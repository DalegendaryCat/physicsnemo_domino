# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains the path definitions for the External Aerodynamics dataset.
"""

from enum import Enum
from pathlib import Path

from constants import DatasetKind


class VTKPaths(str, Enum):
    """Common VTK directory and file patterns."""

    FOAM_DIR = "VTK/simpleFoam_steady_3000"
    INTERNAL = "internal.vtu"
    BOUNDARY = "boundary"

class OpenFoamDatasetPaths:
    """Utility base class for handling OpenFOAM-produced datasets file paths.

    This class provides static methods to construct paths for different components
    of the OpenFOAM dataset such as volume and surface data which are common
    across OpenFOAM datasets.
    """

    @staticmethod
    def _get_index(car_dir: Path) -> str:
        name = car_dir.name
        if not name.startswith("run_"):
            raise ValueError(f"Directory name must start with 'run_', got: {name}")
        return name.removeprefix("run_")

    @staticmethod
    def volume_path(car_dir: Path) -> Path:
        index = OpenFoamDatasetPaths._get_index(car_dir)
        return car_dir / f"volume_{index}.vtu"

    @staticmethod
    def surface_path(car_dir: Path) -> Path:
        index = OpenFoamDatasetPaths._get_index(car_dir)
        return car_dir / f"boundary_{index}.vtp"

class HDBPaths:
    """Utility class for handling HDB dataset file paths.

    This class provides static methods to construct paths for different components
    of the HDB dataset (geometry, volume, and surface data).
    """
    @staticmethod
    def geometry_path(building_dir: Path) -> Path:
        return building_dir / f"{building_dir.name}.stl"
    
    @staticmethod
    def surface_path(building_dir: Path) -> Path:
        return building_dir / f"{building_dir.name}.vtp"
    
    @staticmethod
    def volume_path(building_dir: Path) -> Path:
        return building_dir / f"{building_dir.name}.vtu"

def get_path_getter(kind: DatasetKind):
    """Returns path getter for a given dataset type."""

    match kind:
        case DatasetKind.HDB:
            return HDBPaths
