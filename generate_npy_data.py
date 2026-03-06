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

# conversion from vtk + stl to npy
# need to consider
# surface variables (U, p)
# mesh geometry preservation

# Physical variables
VOLUME_VARS = ["p", "U"]
SURFACE_VARS = ["p", "U", "T"]
GLOBAL_PARAMS_TYPES = {"air_density" : "scalar", "inlet_velocity_x" : "vector", "inlet_velocity_y" : "vector"}
GLOBAL_PARAMS_REFERENCE = {"air_density" : 1.225, "inlet_velocity_x": [0.0], "inlet_velocity_y": [-2.0]}

class OpenFoamSurfaceDataset(Dataset):
    """
    Datapipe for converting OpenFOAM Ahmed Body surface dataset into NumPy arrays.

    This class reads the VTP surface simulation files, STL geometry files, and info
    files containing global parameters (like inlet velocity) to prepare data for
    machine learning workflows.
    """

    def __init__(self, vtp_path: Union[str, Path], vtu_path: Union[str, Path], info_path: Union[str, Path], stl_path: Union[str, Path], surface_variables=None, volume_variables=None, global_params_types=None, global_params_reference=None, device: int = 0):
        """
        Initializes the dataset object.

        Args:
            vtu_path: Path to VTU files (volume CFD results)
            vtp_path: Path to VTP files (surface CFD results).
            info_path: Path to global parameter files (text files).
            stl_path: Path to STL geometry files.
            surface_variables: List of surface fields to extract (default: ["p", "U", "T]).
            volume_variables: List of volume fields (default: ["p", "U", "T"]).
            device: Device ID for loading to GPU (optional).
        """
        self.vtu_path = Path(vtu_path).expanduser()
        self.vtp_path = Path(vtp_path).expanduser()
        self.stl_path = Path(stl_path).expanduser()
        self.info_path = Path(info_path).expanduser()
        assert self.vtp_path.exists(), f"Path {self.vtp_path} does not exist"

        # List all VTP files and shuffle for random sampling
        self.filenames = get_filenames(self.vtp_path)
        random.shuffle(self.filenames)
        
        self.surface_variables = surface_variables or ["p", "U", "T"]
        self.volume_variables = volume_variables or ["p", "U"]
        self.global_params_types = global_params_types
        self.global_params_reference = global_params_reference
        self.device = device

    def __len__(self):
        """Returns the number of files in the dataset."""
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Reads one file and converts it to a dictionary of NumPy arrays.

        Steps:
        1. Read global parameter info (inlet velocity) from info file.
        2. Read STL file to get mesh points, faces, and surface areas.
        3. Read VTP file to get surface CFD fields (pressure, shear stress).
        4. Normalize surface fields using velocity and air density.
        5. Compute surface normals and areas.
        6. Return a dictionary containing all relevant NumPy arrays.
        """
        cfd_filename = self.filenames[idx]
        building_dir = self.vtp_path / cfd_filename

        vtu_path = self.vtu_path / f"{building_dir.stem}.vtu"
        stl_path = self.stl_path / f"{building_dir.stem}.stl"
        info_path = self.info_path / f"{building_dir.stem}"

        # Read inlet velocity from info file
        with open(info_path, "r") as file:
            velocity_x = next(float(line.split("\t")[1].strip(';\n')) for line in file if "windU" in line)
            velocity_y = next(float(line.split("\t")[1].strip(';\n')) for line in file if "windV" in line)
            
        air_density = self.global_params_reference["air_density"]
        # Read STL mesh
        mesh_stl = pv.get_reader(stl_path).read()
        stl_faces = mesh_stl.faces.reshape(-1, 4)[:, 1:]
        stl_sizes = np.array(mesh_stl.compute_cell_sizes(length=False, area=True, volume=False).cell_data["Area"])

        # Read VTP surface data
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(building_dir))
        reader.Update()
        polydata = reader.GetOutput()

        celldata = get_node_to_elem(polydata).GetCellData()
        surface_fields = np.concatenate(get_fields(celldata, self.surface_variables), axis=-1) # / (air_density * velocity**2)

        mesh = pv.PolyData(polydata)
        surface_sizes = np.array(mesh.compute_cell_sizes(length=False, area=True, volume=False).cell_data["Area"])
        surface_normals = mesh.cell_normals / np.linalg.norm(mesh.cell_normals, axis=1)[:, np.newaxis]

        # Read VTU volume data
        vtuReader = vtk.vtkXMLUnstructuredGridReader()
        vtuReader.SetFileName(str(vtu_path))
        vtuReader.Update()
        vtuData = vtuReader.GetOutput()

        vtuMesh = pv.UnstructuredGrid(vtuData)

        volume_mesh_centers, volume_fields = get_volume_data(vtuData, self.volume_variables)
        volume_fields = np.concatenate(volume_fields, axis=-1)


        # Arrange global parameters reference in a list based on the type of the parameter
        global_params_reference_list = []
        for name, type in self.global_params_types.items():
            if type == "vector":
                global_params_reference_list.extend(self.global_params_reference[name])
            elif type == "scalar":
                global_params_reference_list.append(self.global_params_reference[name])
            else:
                raise ValueError(
                    f"Global parameter {name} not supported for this dataset"
                )
        global_params_reference = np.array(
            global_params_reference_list, dtype=np.float32
        )

        # Prepare the list of global parameter values for each simulation file
        global_params_values_list = []
        for key in self.global_params_types.keys():
            if key == "inlet_velocity_x":
                 global_params_values_list.append(velocity_x)
            elif key == "inlet_velocity_y":
                global_params_values_list.append(velocity_y)
            elif key == "air_density":
                 global_params_values_list.append(air_density)
            else:
                raise ValueError(
                    f"Global parameter {key} not supported for this dataset"
                )
        global_params_values = np.array(global_params_values_list, dtype=np.float32)

        return {
            # processed geometry data
            "stl_coordinates": mesh_stl.points.astype(np.float32),
            "stl_centers": mesh_stl.cell_centers().points.astype(np.float32),
            "stl_faces": stl_faces.flatten().astype(np.float32),
            "stl_areas": stl_sizes.astype(np.float32),

            # processed surface data
            "surface_mesh_centers": mesh.cell_centers().points.astype(np.float32),
            "surface_normals": surface_normals.astype(np.float32),
            "surface_areas": surface_sizes.astype(np.float32),
            "surface_fields": surface_fields.astype(np.float32),

            # processed volume data
            "volume_mesh_centers": volume_mesh_centers.astype(np.float32),
            "volume_fields": volume_fields.astype(np.float32),

            "filename": cfd_filename,

            # Global parameters - simulation-wide global quantities used as conditioning inputs
            # for ML models. These capture operating global conditions that affect the entire flow field.

            # global_params_values: Actual values of global parameters for this simulation
            # Example: [stream_velocity, air_density, ...].
            # global_params_reference: Reference/normalization values for `global_params_values`,
            "global_params_values": global_params_values,
            "global_params_reference": global_params_reference,
        }

def process_file(fname: str, fm_data, output_path: str):
    """
    Converts a single VTP/STL file into a .npy file.
    Skips the file if the output already exists or if the input file is missing/empty.
    """
    try:
        full_path = os.path.join(fm_data.vtp_path, fname)
        output_file = os.path.join(output_path, f"{Path(fname).stem}.npy")

        if os.path.exists(output_file):
            return f"- Skipped {fname} (already exists)"
        if not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
            return f"- Skipped {fname} (missing/empty)"

        data = fm_data[fm_data.filenames.index(fname)]
        np.save(output_file, data)
        return f"+ Processed {fname}"
    except Exception as e:
        return f"- Failed {fname}: {e}"

def process_surface_data_batch(dataset_paths: dict, vtu_paths: dict, info_paths: dict, stl_paths: dict, surface_paths: dict):
    """
    Converts all surface data in the dataset into NumPy format and saves them.

    Steps:
    - Ensures output directories exist.
    - Iterates through train/validation/test splits.
    - Loads the dataset using OpenFoamAhmedBodySurfaceDataset.
    - Processes files in parallel using ProcessPoolExecutor.
    - Converts VTP+STL+global velocity into a NumPy dictionary for each case.
    - Saves the .npy files in the corresponding prepared surface data folder.
    """
    for path in surface_paths.values(): os.makedirs(path, exist_ok=True)

    print("=== Starting Processing ===")
    for key, dataset_path in dataset_paths.items():
        surface_path = surface_paths[key]
        os.makedirs(surface_path, exist_ok=True)
        fm_data = OpenFoamSurfaceDataset(dataset_path, vtu_paths[key], info_paths[key], stl_paths[key], VOLUME_VARS, SURFACE_VARS,GLOBAL_PARAMS_TYPES,GLOBAL_PARAMS_REFERENCE)
        file_list = [fname for fname in fm_data.filenames if fname.endswith(".vtp")]

        print(f"\nProcessing {len(file_list)} files from {dataset_path} → {surface_path}...")

        with ProcessPoolExecutor() as executor:
            for msg in tqdm(
                executor.map(process_file, file_list, [fm_data]*len(file_list), [surface_path]*len(file_list)),
                total=len(file_list),
                desc=f"Processing {key}",
                dynamic_ncols=True
            ):
                if msg and msg.startswith("-"):
                    print(msg)

    print("=== All Processing Completed Successfully ===")