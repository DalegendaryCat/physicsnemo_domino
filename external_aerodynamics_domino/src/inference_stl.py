# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Inference script for DoMINO on HDB dataset. Only requires STL files.
If a .vtu file is present alongside the STL, its mesh coordinates are used
for volume inference so predictions align with the CFD mesh. Falls back to
random sampling if no VTU is found.

Surface fields: [p, Ux, Uy, Uz]   (p=scalar idx 0, U=vector idx 1-3)
Volume  fields: [Ux, Uy, Uz, p]   (U=vector idx 0-2, p=scalar idx 3)
Forces (pressure only, no WSS):
  Drag(X) = sum(p * nx * A)
  Lift(Z) = sum(p * nz * A)
  Side(Y) = sum(p * ny * A)
"""

import os
import time

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import vtk
from vtk.util import numpy_support
import pyvista as pv
import torchinfo

from physicsnemo.utils.memory import unified_gpu_memory
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataPipe
from physicsnemo.models.domino.model import DoMINO
from physicsnemo.utils.domino.utils import sample_points_on_mesh
from physicsnemo.utils.domino.vtk_file_utils import write_to_vtp, write_to_vtu
from physicsnemo.utils.profiling import Profiler

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

from utils import coordinate_distributed_environment, load_scaling_factors, get_num_vars
from loss import compute_loss_dict

nvmlInit()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sample_volume_points(c_min, c_max, n_points, device, eps=1e-7):
    """Uniform random points inside bounding box."""
    u = torch.rand(n_points, 3, device=device, dtype=torch.float32) * (1 - 2*eps) + eps
    return (c_max - c_min) * u + c_min


def load_stl_to_tensors(stl_path, device):
    """Read STL → (vertices [V,3], faces_flat [F*3]) on device."""
    mesh = pv.get_reader(stl_path).read()
    vertices = np.array(mesh.points, dtype=np.float32)
    faces    = np.array(mesh.faces).reshape((-1, 4))[:, 1:]
    stl_coordinates = torch.from_numpy(vertices).to(torch.float32).to(device)
    stl_faces       = torch.from_numpy(faces.flatten()).to(torch.int32).to(device)
    return stl_coordinates, stl_faces


def load_vtu_coords(vtu_path, device):
    """
    Read cell centre coordinates from a VTU file.
    Uses pyvista cell_centers() to match what the PhysicsNeMo curator extracts.
    Returns tensor [N, 3] on device, or None if file not found.
    """
    if not os.path.exists(vtu_path):
        return None
    mesh = pv.read(vtu_path)
    centres = mesh.cell_centers().points   # [n_cells, 3]
    return torch.from_numpy(centres.astype(np.float32)).to(device)


def build_global_params(cfg, device):
    """
    Build global_params_values and global_params_reference as [n_params, 1]
    tensors so after unsqueeze(0) the model receives [1, n_params, 1].
    """
    names = list(cfg.variables.global_parameters.keys())
    types = {n: cfg.variables.global_parameters[n]["type"]      for n in names}
    refs  = {n: cfg.variables.global_parameters[n]["reference"] for n in names}

    flat = []
    for n, t in types.items():
        if t == "vector":
            flat.extend(refs[n])
        else:
            flat.append(refs[n])

    t = torch.tensor(flat, dtype=torch.float32, device=device).unsqueeze(-1)
    return t, t.clone()


def stl_geometry(stl_coordinates, stl_faces):
    """Compute face centers, unit normals, and areas from STL tensors."""
    tri      = stl_coordinates[stl_faces.reshape((-1, 3))]
    centers  = tri.mean(dim=1)
    d1       = tri[:, 1] - tri[:, 0]
    d2       = tri[:, 2] - tri[:, 0]
    cross    = torch.linalg.cross(d1, d2, dim=1)
    norm_len = torch.linalg.norm(cross, dim=1)
    normals  = cross / norm_len.unsqueeze(1)
    areas    = 0.5 * norm_len
    return centers, normals, areas


# ─────────────────────────────────────────────────────────────────────────────
# Core inference
# ─────────────────────────────────────────────────────────────────────────────

def run_surface_batch(datapipe, model, stl_coordinates, stl_faces,
                      stl_centers, stl_normals, stl_areas,
                      surface_centers, surface_normals, surface_areas,
                      global_params_values, global_params_reference):
    """Run one surface batch through datapipe → model → unscale."""
    device = stl_coordinates.device
    d = {
        "stl_coordinates":         stl_coordinates,
        "stl_faces":               stl_faces,
        "stl_centers":             stl_centers,
        "stl_areas":               stl_areas,
        "global_params_values":    global_params_values,
        "global_params_reference": global_params_reference,
        "surface_mesh_centers":    surface_centers,
        "surface_normals":         surface_normals,
        "surface_areas":           surface_areas,
        "surface_faces":           stl_faces,
    }
    if datapipe.model_type == "combined":
        c_min = datapipe.config.bounding_box_dims[1]
        c_max = datapipe.config.bounding_box_dims[0]
        d["volume_mesh_centers"] = sample_volume_points(
            c_min, c_max, surface_centers.shape[0], device)

    pre = datapipe.process_data(d)
    pre = {k: v.unsqueeze(0) for k, v in pre.items()}

    with torch.no_grad():
        _, out_surf = model(pre)
    _, out_surf = datapipe.unscale_model_outputs(None, out_surf)
    return out_surf


def inference_on_single_stl(stl_coordinates, stl_faces,
                             global_params_values, global_params_reference,
                             model, datapipe, batch_size,
                             vol_coords_input=None,   # [N,3] from VTU or None
                             total_points=1_240_000,
                             gpu_handle=None, logger=None):
    """
    Full inference on one STL.

    Volume points source (in priority order):
      1. vol_coords_input — actual CFD mesh coordinates loaded from .vtu
      2. random sampling across the bounding box (fallback)

    Returns:
        stl_center_results : [1, n_faces, n_surf_vars]
        volume_results     : [1, N, n_vol_vars]
        volume_coords      : [N, 3]  coordinates matching volume_results
        stl_centers, stl_normals, stl_areas
    """
    device = stl_coordinates.device
    t0     = time.perf_counter()

    stl_centers, stl_normals, stl_areas = stl_geometry(stl_coordinates, stl_faces)

    # ── Determine volume coordinate source ───────────────────────────────────
    if vol_coords_input is not None:
        # Use actual VTU mesh coords — chunk through them sequentially
        all_vol_coords = vol_coords_input
        n_total        = all_vol_coords.shape[0]
        use_vtu_coords = True
        (logger.info if logger else print)(
            f"  Using VTU mesh coords: {n_total} points")
    else:
        all_vol_coords = None
        n_total        = total_points
        use_vtu_coords = False
        (logger.info if logger else print)(
            f"  No VTU found — random volume sampling: {n_total} points")

    # ── 1. Volume + surface random-sample inference loop ─────────────────────
    volume_results = []
    volume_coords  = []
    processed      = 0

    while processed < n_total:
        inner_t = time.perf_counter()

        # Volume coords for this batch
        if use_vtu_coords:
            end     = min(processed + batch_size, n_total)
            vol_pts = all_vol_coords[processed:end]
        else:
            end     = processed + batch_size
            c_min   = datapipe.config.bounding_box_dims[1]
            c_max   = datapipe.config.bounding_box_dims[0]
            vol_pts = sample_volume_points(c_min, c_max, batch_size, device)

        volume_coords.append(vol_pts)

        d = {
            "stl_coordinates":         stl_coordinates,
            "stl_faces":               stl_faces,
            "stl_centers":             stl_centers,
            "stl_areas":               stl_areas,
            "global_params_values":    global_params_values,
            "global_params_reference": global_params_reference,
        }

        if datapipe.model_type in ("surface", "combined"):
            pts, faces, areas, norms = sample_points_on_mesh(
                stl_coordinates, stl_faces, vol_pts.shape[0],
                mesh_normals=stl_normals, mesh_areas=stl_areas)
            d["surface_mesh_centers"] = pts
            d["surface_normals"]      = norms
            d["surface_areas"]        = areas
            d["surface_faces"]        = faces

        if datapipe.model_type in ("volume", "combined"):
            d["volume_mesh_centers"] = vol_pts

        pre = datapipe.process_data(d)
        pre = {k: v.unsqueeze(0) for k, v in pre.items()}

        with torch.no_grad():
            out_vol, _ = model(pre)
        out_vol, _ = datapipe.unscale_model_outputs(out_vol, None)

        volume_results.append(out_vol)
        processed = end

        now = time.perf_counter()
        log = (f"  {processed}/{n_total} pts | "
               f"elapsed {now-t0:.1f}s | "
               f"{vol_pts.shape[0]/(now-inner_t):.0f} pts/s")
        if gpu_handle:
            log += f" | GPU {nvmlDeviceGetMemoryInfo(gpu_handle).used/(1024**3):.2f}GB"
        (logger.info if logger else print)(log)

    # ── 2. STL-face-centre inference (ALL faces in batches) ──────────────────
    if datapipe.model_type in ("surface", "combined"):
        n_faces          = stl_centers.shape[0]
        stl_surf_chunks  = []

        for start in range(0, n_faces, batch_size):
            end        = min(start + batch_size, n_faces)
            chunk_surf = run_surface_batch(
                datapipe, model,
                stl_coordinates, stl_faces,
                stl_centers, stl_normals, stl_areas,
                stl_centers[start:end],
                stl_normals[start:end],
                stl_areas[start:end],
                global_params_values,
                global_params_reference,
            )
            stl_surf_chunks.append(chunk_surf)

        stl_center_results = torch.cat(stl_surf_chunks, dim=1)  # [1, n_faces, n_vars]
    else:
        stl_center_results = None

    # ── 3. Stack results ──────────────────────────────────────────────────────
    volume_results = (torch.cat(volume_results, dim=1)
                      if volume_results and all(v is not None for v in volume_results)
                      else None)
    volume_coords  = (torch.cat(volume_coords, dim=0)
                      if volume_coords else None)   # [N, 3]

    return stl_center_results, volume_results, volume_coords, stl_centers, stl_normals, stl_areas


# ─────────────────────────────────────────────────────────────────────────────
# Epoch loop
# ─────────────────────────────────────────────────────────────────────────────

def inference_epoch(dirnames, input_path, datapipe, model,
                    gpu_handle, logger, cfg,
                    surface_variable_names, volume_variable_names,
                    device, batch_size, total_points):

    surf_p_name = surface_variable_names[0]   # "p"
    surf_U_name = surface_variable_names[1]   # "U"
    vol_U_name  = volume_variable_names[0]    # "U"
    vol_p_name  = volume_variable_names[1]    # "p"

    save_path = cfg.eval.save_path
    os.makedirs(save_path, exist_ok=True)

    for i_batch, dirname in enumerate(dirnames):
        try:
            t_load = time.perf_counter()
            stl_path = os.path.join(input_path, dirname, f"{dirname}.stl")
            vtu_path_input = os.path.join(input_path, dirname, f"{dirname}.vtu")

            logger.info(f"[{i_batch}] Loading {stl_path}")
            stl_coordinates, stl_faces = load_stl_to_tensors(stl_path, device)

            # Load VTU mesh coordinates if available — gives proper point distribution
            vol_coords_input = load_vtu_coords(vtu_path_input, device)
            if vol_coords_input is not None:
                logger.info(f"[{i_batch}] VTU mesh loaded: {vol_coords_input.shape[0]} pts")
            else:
                logger.info(f"[{i_batch}] No VTU found — will use random sampling")

            global_params_values, global_params_reference = build_global_params(cfg, device)

            logger.info(f"[{i_batch}] Load time {time.perf_counter()-t_load:.2f}s  "
                        f"verts={stl_coordinates.shape[0]}  "
                        f"faces={stl_faces.shape[0]//3}")

            t0 = time.perf_counter()
            stl_center_results, volume_results, volume_coords, \
            stl_centers, stl_normals, stl_areas = inference_on_single_stl(
                stl_coordinates, stl_faces,
                global_params_values, global_params_reference,
                model, datapipe, batch_size,
                vol_coords_input=vol_coords_input,
                total_points=total_points,
                gpu_handle=gpu_handle,
                logger=logger,
            )
            logger.info(f"[{i_batch}] inference {time.perf_counter()-t0:.2f}s")

            # ── Forces ───────────────────────────────────────────────────────
            if stl_center_results is not None:
                pred    = stl_center_results[0]   # [n_faces, 4]
                pred_p  = pred[:, 0]
                pred_Ux = pred[:, 1]

                drag = torch.sum(pred_p * stl_normals[:, 0] * stl_areas)
                lift = torch.sum(pred_p * stl_normals[:, 2] * stl_areas)
                side = torch.sum(pred_p * stl_normals[:, 1] * stl_areas)
                logger.info(f"[{dirname}]  "
                            f"Drag(X)={drag.item():.2f}  "
                            f"Lift(Z)={lift.item():.2f}  "
                            f"Side(Y)={side.item():.2f}")
                logger.info(f"[{dirname}]  "
                            f"surf p [{pred_p.min().item():.3f}, {pred_p.max().item():.3f}]  "
                            f"Ux [{pred_Ux.min().item():.3f}, {pred_Ux.max().item():.3f}]")

            if volume_results is not None:
                vol = volume_results[0]
                logger.info(f"[{dirname}]  "
                            f"vol Ux [{vol[:,0].min().item():.3f}, {vol[:,0].max().item():.3f}]  "
                            f"p [{vol[:,3].min().item():.3f}, {vol[:,3].max().item():.3f}]")

            # ── Save surface VTP ──────────────────────────────────────────────
            vtp_path = os.path.join(save_path, f"boundary_{dirname}_predicted.vtp")
            vtu_path_out = os.path.join(save_path, f"volume_{dirname}_predicted.vtu")

            if stl_center_results is not None:
                stl_coords_np = stl_coordinates.cpu().numpy()
                stl_faces_np  = stl_faces.cpu().numpy().reshape((-1, 3))
                n_faces       = stl_faces_np.shape[0]

                faces_pv = np.hstack([
                    np.full((n_faces, 1), 3, dtype=np.int64),
                    stl_faces_np.astype(np.int64),
                ]).flatten()
                mesh_pv  = pv.PolyData(stl_coords_np, faces_pv)
                vtk_poly = mesh_pv.cast_to_unstructured_grid().extract_surface()

                pred_np = stl_center_results[0].cpu().numpy()   # [n_faces, 4]

                arr_p = numpy_support.numpy_to_vtk(np.ascontiguousarray(pred_np[:, 0:1]))
                arr_p.SetName(f"{surf_p_name}Pred")
                vtk_poly.GetCellData().AddArray(arr_p)

                arr_U = numpy_support.numpy_to_vtk(np.ascontiguousarray(pred_np[:, 1:4]))
                arr_U.SetName(f"{surf_U_name}Pred")
                vtk_poly.GetCellData().AddArray(arr_U)

                write_to_vtp(vtk_poly, vtp_path)
                logger.info(f"Saved surface → {vtp_path}")

            # ── Save volume VTU ───────────────────────────────────────────────
            if volume_results is not None and vol_coords_input is not None:
                vol_np = volume_results[0].cpu().numpy()   # [N, 4]
                n_pred = vol_np.shape[0]

                # Load original VTU — use pyvista to extract only the N cells
                # that match our predictions (first N cells by index, same order
                # as the coords we loaded with load_vtu_coords)
                mesh_pv = pv.read(vtu_path_input)
                n_cells = mesh_pv.n_cells

                if n_pred == n_cells:
                    # Perfect match — append directly as cell data
                    mesh_pv.cell_data[f"{vol_U_name}Pred"] = vol_np[:, 0:3]
                    mesh_pv.cell_data[f"{vol_p_name}Pred"] = vol_np[:, 3:4].squeeze(-1)
                    mesh_pv.save(vtu_path_out)
                    logger.info(f"Saved volume  → {vtu_path_out}  ({n_pred} cells, full mesh)")
                else:
                    # Partial — extract just the cells we predicted on
                    subset = mesh_pv.extract_cells(np.arange(n_pred))
                    subset.cell_data[f"{vol_U_name}Pred"] = vol_np[:, 0:3]
                    subset.cell_data[f"{vol_p_name}Pred"] = vol_np[:, 3:4].squeeze(-1)
                    subset.save(vtu_path_out)
                    logger.info(f"Saved volume  → {vtu_path_out}  ({n_pred}/{n_cells} cells, partial mesh)")

                # Verify what was written
                verify = pv.read(vtu_path_out)
                logger.info(f"  Verify — cell arrays: {list(verify.cell_data.keys())}")
                logger.info(f"  Verify — point arrays: {list(verify.point_data.keys())}")

        except Exception as e:
            import traceback
            logger.info(f"[{dirname}] SKIPPED — {e}")
            logger.info(traceback.format_exc())
            continue


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    DistributedManager.initialize()
    dist = DistributedManager()
    coordinate_distributed_environment(cfg)

    nvmlInit()
    gpu_handle = nvmlDeviceGetHandleByIndex(dist.device.index)

    logger = PythonLogger("Inference")
    logger = RankZeroLoggingWrapper(logger, dist)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    model_type = cfg.model.model_type
    num_vol_vars, num_surf_vars, num_global_features = get_num_vars(cfg, model_type)

    surface_variable_names = (list(cfg.variables.surface.solution.keys())
                               if model_type in ("combined", "surface") else [])
    volume_variable_names  = (list(cfg.variables.volume.solution.keys())
                               if model_type in ("combined", "volume")  else [])

    batch_size = cfg.model.volume_points_sample

    vol_factors, surf_factors = load_scaling_factors(cfg)

    datapipe = DoMINODataPipe(
        input_path=cfg.eval.test_path,
        model_type=cfg.model.model_type,
        pin_memory=False,
        phase="test",
        volume_factors=vol_factors,
        surface_factors=surf_factors,
        scaling_type=cfg.model.normalization,
        bounding_box_dims=cfg.data.bounding_box,
        bounding_box_dims_surf=cfg.data.bounding_box_surface,
        grid_resolution=cfg.model.interp_res,
        normalize_coordinates=cfg.data.normalize_coordinates,
        # Disable — we supply exactly the points we want ourselves
        sample_in_bbox=False,
        sampling=False,
        gpu_preprocessing=cfg.data.gpu_preprocessing,
        gpu_output=cfg.data.gpu_output,
        surface_points_sample=cfg.model.surface_points_sample,
        volume_points_sample=cfg.model.volume_points_sample,
        geom_points_sample=cfg.model.geom_points_sample,
        num_surface_neighbors=cfg.model.num_neighbors_surface,
    )

    model = DoMINO(
        input_features=3,
        output_features_vol=num_vol_vars,
        output_features_surf=num_surf_vars,
        global_features=num_global_features,
        model_parameters=cfg.model,
    ).to(dist.device)

    logger.info(f"Model:\n{torchinfo.summary(model, verbose=0, depth=2)}\n")

    if dist.world_size > 1:
        torch.distributed.barrier()

    checkpoint = torch.load(
        to_absolute_path(os.path.join(cfg.resume_dir, cfg.eval.checkpoint_name)),
        map_location=dist.device,
    )
    model.load_state_dict(checkpoint)
    print("Model loaded")
    model.eval()

    input_path = cfg.eval.test_path
    dirnames   = sorted([
        d for d in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, d)) and not d.startswith(".")
    ])
    logger.info(f"Found {len(dirnames)} test cases: {dirnames}")

    t0 = time.perf_counter()
    with Profiler():
        inference_epoch(
            dirnames=dirnames,
            input_path=input_path,
            datapipe=datapipe,
            model=model,
            gpu_handle=gpu_handle,
            logger=logger,
            cfg=cfg,
            surface_variable_names=surface_variable_names,
            volume_variable_names=volume_variable_names,
            device=dist.device,
            batch_size=batch_size,
            total_points=cfg.eval.num_points,
        )
    logger.info(f"Total time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()