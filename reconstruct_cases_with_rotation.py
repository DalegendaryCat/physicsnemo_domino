#!/usr/bin/env python3
# reconstruct_cases_with_rotation.py
# Reconstructs all OpenFOAM cases, converts to VTK, copies STL, and organises
# into train/test/validation split. Also runs file format conversion to produce
# .vtu and .vtp files ready for the ETL pipeline.
#
# Usage: python3 reconstruct_cases_with_rotation.py /path/to/cases/directory

import os
import subprocess
import sys
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import vtk
from physicsnemo.utils.domino.vtk_file_utils import write_to_vtp, write_to_vtu

# ─── Config ──────────────────────────────────────────────────────────────────

TIME = "1250"
RECONSTRUCT_OPTS = ["-time", TIME]
VTK_OPTS = ["-time", TIME]
OUTPUT_DIR = "/home/nguye/physicsnemo/Dataset/hdb_input_rotated"
SPLIT = {"train": 0.8, "test": 0.1, "validation": 0.1}
RANDOM_SEED = 42        # Set to None for a different split every run
MAX_WORKERS = 4         # Number of cases to process in parallel

# Rotation angles to align all cases to North direction
ROTATION_MAP = {
    "N": 0,    # no rotation needed
    "S": 180,  # rotate 180°
    "E": 90,   # rotate 90°
    "W": 270,  # rotate 270°
}

# ─── OpenFOAM helpers ────────────────────────────────────────────────────────

def is_openfoam_case(path):
    has_system = os.path.isdir(os.path.join(path, "system"))
    has_processors = any(
        d.startswith("processor") for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    )
    return has_system, has_processors


def run_command(cmd, log_path):
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return result.returncode == 0


def find_vtk_file(case_path, case_name):
    vtk_dir = os.path.join(case_path, "VTK")
    if not os.path.isdir(vtk_dir):
        return None
    for f in os.listdir(vtk_dir):
        if f.endswith(".vtk") and f.startswith(case_name):
            return os.path.join(vtk_dir, f)
    return None


def find_buildings_vtk_file(case_path):
    """Find the buildings patch VTK file in the VTK/buildings subdirectory."""
    buildings_dir = os.path.join(case_path, "VTK", "buildings")
    if not os.path.isdir(buildings_dir):
        return None
    for f in os.listdir(buildings_dir):
        if f.endswith(".vtk"):
            return os.path.join(buildings_dir, f)
    return None


# ─── File format conversion (from file_format_converter.py) ──────────────────

def convert_vtk_to_vtu(vtk_filename: str, vtu_filename: str) -> None:
    """Convert full domain .vtk (UnstructuredGrid) to .vtu."""
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_filename)
    reader.Update()
    if not reader.GetOutput():
        print(f"[ERROR] Failed to read {vtk_filename}")
        return
    write_to_vtu(reader.GetOutput(), vtu_filename)
    del reader


def convert_buildings_vtk_to_vtp(buildings_vtk: str, vtp_filename: str) -> None:
    """Convert buildings patch .vtk (PolyData) to .vtp."""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(buildings_vtk)
    reader.Update()
    if not reader.GetOutput() or reader.GetOutput().GetNumberOfPoints() == 0:
        print(f"[ERROR] Failed to read or empty: {buildings_vtk}")
        return
    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputConnection(reader.GetOutputPort())
    clean_filter.Update()
    write_to_vtp(clean_filter.GetOutput(), vtp_filename)
    del reader


def convert_vtp_to_stl(vtp_filename: str, stl_filename: str) -> None:
    """Convert buildings .vtp (PolyData) to .stl."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_filename)
    reader.Update()
    if not reader.GetOutput() or reader.GetOutput().GetNumberOfPoints() == 0:
        print(f"[ERROR] Failed to read or empty: {vtp_filename}")
        return
    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputConnection(reader.GetOutputPort())
    triangulate.Update()
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(stl_filename)
    writer.SetInputConnection(triangulate.GetOutputPort())
    writer.Write()
    del reader, writer


# ─── Per-case processing ─────────────────────────────────────────────────────

def process_case(args):
    """Reconstruct and convert a single OpenFOAM case. Runs in a worker process."""
    cases_dir, case_name = args
    case_path = os.path.join(cases_dir, case_name)
    status = {"case": case_name, "success": False, "error": None}

    # reconstructPar
    log_path = os.path.join(case_path, "reconstructPar.log")
    cmd = ["reconstructPar"] + RECONSTRUCT_OPTS + ["-case", case_path]
    if not run_command(cmd, log_path):
        status["error"] = "reconstructPar failed"
        return status

    # Rotate case to North direction based on wind direction in filename
    direction = case_name.split("_")[-1].upper()
    rotation_angle = ROTATION_MAP.get(direction, 0)
    if rotation_angle != 0:
        rotate_log_path = os.path.join(case_path, "transformPoints.log")
        rotate_cmd = [
            "transformPoints",
            f"Rz={rotation_angle}",
            "-rotateFields",
            "-case", case_path,
        ]
        if not run_command(rotate_cmd, rotate_log_path):
            status["error"] = f"transformPoints failed (Rz={rotation_angle})"
            return status

    # foamToVTK
    vtk_log_path = os.path.join(case_path, "foamToVTK.log")
    vtk_cmd = ["foamToVTK"] + VTK_OPTS + ["-case", case_path]
    if not run_command(vtk_cmd, vtk_log_path):
        status["error"] = "foamToVTK failed"
        return status

    status["success"] = True
    return status


# ─── Split helper ────────────────────────────────────────────────────────────

def split_cases(case_names):
    cases = list(case_names)
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    random.shuffle(cases)
    n = len(cases)
    n_train = round(n * SPLIT["train"])
    n_test = round(n * SPLIT["test"])
    return {
        "train": cases[:n_train],
        "test": cases[n_train:n_train + n_test],
        "validation": cases[n_train + n_test:]
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    DEFAULT_CASES_DIR = "/home/nguye/physicsnemo/simulation_data/simulation_data"

    if len(sys.argv) < 2:
        cases_dir = DEFAULT_CASES_DIR
        print(f"No path provided — using default: {cases_dir}")
    else:
        cases_dir = os.path.abspath(sys.argv[1])

    total_start = time.perf_counter()

    print("============================================")
    print(" OpenFOAM Batch Reconstruction + VTK Export")
    print(f" Cases dir:    {cases_dir}")
    print(f" Output dir:   {OUTPUT_DIR}")
    print(f" Time:         {TIME}")
    print(f" Workers:      {MAX_WORKERS}")
    print(f" Split:        train={int(SPLIT['train']*100)}% / test={int(SPLIT['test']*100)}% / validation={int(SPLIT['validation']*100)}%")
    print("============================================\n")

    if not os.path.isdir(cases_dir):
        print(f"Error: Cases directory '{cases_dir}' not found.")
        sys.exit(1)

    # Create output directories
    for split in SPLIT:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

    # Find valid cases
    all_subdirs = sorted([
        d for d in os.listdir(cases_dir)
        if os.path.isdir(os.path.join(cases_dir, d))
    ])

    valid_cases = []
    skipped = []
    for case_name in all_subdirs:
        case_path = os.path.join(cases_dir, case_name)
        has_system, has_processors = is_openfoam_case(case_path)
        if not has_system or not has_processors:
            reason = "no system/ folder" if not has_system else "no processor* dirs"
            print(f"Skipping '{case_name}' — {reason}")
            skipped.append(case_name)
        else:
            valid_cases.append(case_name)

    print(f"\nFound {len(valid_cases)} valid cases — processing with {MAX_WORKERS} workers...\n")

    # ── Parallel reconstruct + foamToVTK ──────────────────────────────────────
    recon_start = time.perf_counter()
    success = []
    failed = []

    args_list = [(cases_dir, case_name) for case_name in valid_cases]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_case, args): args[1] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            case_name = result["case"]
            if result["success"]:
                print(f"  + {case_name}")
                success.append(case_name)
            else:
                print(f"  x {case_name} — {result['error']}")
                failed.append(case_name)

    if not success:
        print("\nNo cases succeeded — skipping dataset organisation.")
        sys.exit(1)

    recon_elapsed = time.perf_counter() - recon_start
    print(f"\n  Reconstruction completed in {recon_elapsed:.1f}s ({recon_elapsed/60:.1f} min)")

    # ── Split & Organise ──────────────────────────────────────────────────────
    print("\n============================================")
    print(" Organising into train/test/validation")
    print("============================================\n")

    organise_start = time.perf_counter()
    splits = split_cases(success)

    for split_name, case_list in splits.items():
        print(f"{split_name.upper()} ({len(case_list)} cases)")
        for case_name in case_list:
            case_start = time.perf_counter()
            case_path = os.path.join(cases_dir, case_name)
            dest_case_dir = os.path.join(OUTPUT_DIR, split_name, case_name)
            os.makedirs(dest_case_dir, exist_ok=True)

            # ── 1. Convert buildings VTK -> VTP ───────────────────────────────
            buildings_vtk = find_buildings_vtk_file(case_path)
            if buildings_vtk:
                vtp_dest = os.path.join(dest_case_dir, f"{case_name}.vtp")
                convert_buildings_vtk_to_vtp(buildings_vtk, vtp_dest)
                print(f"  + {case_name}_buildings.vtk -> {case_name}.vtp")
            else:
                print(f"  - No buildings VTK found for '{case_name}'")
                vtp_dest = None

            # ── 2. Convert VTP -> STL (higher resolution than triSurface STL) ──
            if vtp_dest and os.path.exists(vtp_dest):
                stl_dest = os.path.join(dest_case_dir, f"{case_name}.stl")
                convert_vtp_to_stl(vtp_dest, stl_dest)
                print(f"  + {case_name}.vtp -> {case_name}.stl")
            else:
                print(f"  - Skipping STL conversion — no VTP available for '{case_name}'")

            # ── 3. Convert case VTK -> VTU ────────────────────────────────────
            vtk_file = find_vtk_file(case_path, case_name)
            if vtk_file:
                vtu_dest = os.path.join(dest_case_dir, f"{case_name}.vtu")
                convert_vtk_to_vtu(vtk_file, vtu_dest)
                print(f"  + {case_name}.vtk -> {case_name}.vtu")
            else:
                print(f"  x VTK file not found for '{case_name}'")

            case_elapsed = time.perf_counter() - case_start
            print(f"  → {case_name} done in {case_elapsed:.1f}s")

        print()

    organise_elapsed = time.perf_counter() - organise_start
    total_elapsed = time.perf_counter() - total_start

    # ── Summary ───────────────────────────────────────────────────────────────
    print("============================================")
    print(" Summary")
    print("============================================")
    print(f" Succeeded: {len(success)}")
    for c in success:
        print(f"   + {c}")
    print(f" Failed:    {len(failed)}")
    for c in failed:
        print(f"   x {c}")
    print(f" Skipped:   {len(skipped)}")
    for c in skipped:
        print(f"   - {c}")
    print("============================================")
    print(f"\n Timing breakdown:")
    print(f"   Reconstruction + rotate + foamToVTK : {recon_elapsed:.1f}s ({recon_elapsed/60:.1f} min)")
    print(f"   Organise + convert         : {organise_elapsed:.1f}s ({organise_elapsed/60:.1f} min)")
    print(f"   Total                      : {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("============================================")
    print(f"\nOutput structure per case:")
    print(f"  {OUTPUT_DIR}/<split>/<case_name>/")
    print(f"    <case_name>.stl   (converted from VTK/buildings/<>.vtk via VTP)")
    print(f"    <case_name>.vtp   (converted from VTK/buildings/<>.vtk)")
    print(f"    <case_name>.vtu   (converted from VTK/<case_name>.vtk)")
    print(f"\nDataset saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()