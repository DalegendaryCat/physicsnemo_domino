# importing relevant libraries
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Union
import numpy as np
import pyvista as pv
import vtk
from stl import mesh
from tqdm import tqdm
from physicsnemo.utils.domino.utils import *
from physicsnemo.utils.domino.vtk_file_utils import *
from torch.utils.data import Dataset


def convert_vtk_to_stl(vtk_filename: str, stl_filename: str) -> None:
    """Converts a single .vtk file to .stl format."""
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_filename)
    reader.Update()

    if not reader.GetOutput():
        print(f"[ERROR] Failed to read {vtk_filename}")
        return

    # Extract surface
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(reader.GetOutput())
    surface_filter.Update()

    if not surface_filter.GetOutput():
        print(f"[ERROR] Surface extraction failed for {vtk_filename}")
        return

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(stl_filename)
    writer.SetInputConnection(surface_filter.GetOutputPort())
    writer.Write()
    del reader, writer


def convert_vtk_to_vtp(vtk_filename: str, vtp_filename: str) -> None:
    """Converts a single .vtk file to .vtp format using write_to_vtp."""
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_filename)
    reader.Update()

    if not reader.GetOutput():
        print(f"[ERROR] Failed to read {vtk_filename}")
        return

    # Extract surface
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputConnection(reader.GetOutputPort())
    surface_filter.Update()

    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputConnection(surface_filter.GetOutputPort())
    clean_filter.Update()

    write_to_vtp(clean_filter.GetOutput(), vtp_filename)
    del reader


def convert_vtk_to_vtu(vtk_filename: str, vtu_filename: str) -> None:
    """Converts a single .vtk file to .vtu format using write_to_vtu."""
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_filename)
    reader.Update()

    if not reader.GetOutput():
        print(f"[ERROR] Failed to read {vtk_filename}")
        return

    write_to_vtu(reader.GetOutput(), vtu_filename)
    del reader


def process_stl_file(vtk_file: str, output_path: str) -> None:
    """Processes a single .vtk file and saves it as .stl."""
    output_file = os.path.join(output_path, os.path.basename(vtk_file).replace(".vtk", ".stl"))
    convert_vtk_to_stl(vtk_file, output_file)


def process_vtp_file(vtk_file: str, output_path: str) -> None:
    """Processes a single .vtk file and saves it as .vtp."""
    output_file = os.path.join(output_path, os.path.basename(vtk_file).replace(".vtk", ".vtp"))
    convert_vtk_to_vtp(vtk_file, output_file)


def process_vtu_file(vtk_file: str, output_path: str) -> None:
    """Processes a single .vtk file and saves it as .vtu."""
    output_file = os.path.join(output_path, os.path.basename(vtk_file).replace(".vtk", ".vtu"))
    convert_vtk_to_vtu(vtk_file, output_file)


def convert_vtk_to_stl_vtp_vtu_batch(
    dataset_paths: dict,
    stl_paths: dict,
    vtp_paths: dict,
    vtu_paths: dict,
) -> None:
    """Processes all .vtk files and saves them as .stl, .vtp, and .vtu in their respective paths."""
    print("\n=== Starting Conversion Process ===")

    for path in stl_paths.values():
        os.makedirs(path, exist_ok=True)
    for path in vtp_paths.values():
        os.makedirs(path, exist_ok=True)
    for path in vtu_paths.values():
        os.makedirs(path, exist_ok=True)

    for key, dataset_path in dataset_paths.items():
        stl_path = stl_paths[key]
        vtp_path = vtp_paths[key]
        vtu_path = vtu_paths[key]

        vtk_files = [
            os.path.join(dataset_path, f)
            for f in os.listdir(dataset_path)
            if f.endswith(".vtk")
        ]

        if not vtk_files:
            print(f"[WARNING] No .vtk files found in {dataset_path}")
            continue

        print(f"\nProcessing {len(vtk_files)} files from {dataset_path} → {stl_path}...")
        with ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(process_stl_file, vtk_files, [stl_path] * len(vtk_files)),
                      total=len(vtk_files), desc=f"Converting {key} to STL", dynamic_ncols=True))

        print(f"\nProcessing {len(vtk_files)} files from {dataset_path} → {vtp_path}...")
        with ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(process_vtp_file, vtk_files, [vtp_path] * len(vtk_files)),
                      total=len(vtk_files), desc=f"Converting {key} to VTP", dynamic_ncols=True))

        print(f"\nProcessing {len(vtk_files)} files from {dataset_path} → {vtu_path}...")
        with ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(process_vtu_file, vtk_files, [vtu_path] * len(vtk_files)),
                      total=len(vtk_files), desc=f"Converting {key} to VTU", dynamic_ncols=True))

    print("\n=== All Conversions Completed Successfully ===")