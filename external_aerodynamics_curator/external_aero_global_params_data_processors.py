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

import logging

import numpy as np
from schemas import ExternalAerodynamicsExtractedDataInMemory

logging.basicConfig(
    format="%(asctime)s - Process %(process)d - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def default_global_params_processing_for_external_aerodynamics(
    data: ExternalAerodynamicsExtractedDataInMemory,
    global_parameters: dict,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Default global parameters processing for External Aerodynamics.

    Extracts and flattens global parameter references from config into a 1D numpy array.
    Handles both vector and scalar parameter types.

    Args:
        data: Container with simulation data and metadata
        global_parameters: Dict from config with structure:
            {
                "param_name": {
                    "type": "vector" or "scalar",
                    "reference": value or list
                }
            }

    Returns:
        Updated `data` with global_params_reference set
    """

    # Build dictionaries for types and reference values
    global_params_types = {
        name: params["type"] for name, params in global_parameters.items()
    }

    global_params_reference_dict = {
        name: params["reference"] for name, params in global_parameters.items()
    }

    # Arrange global parameters reference in a list based on the type of the parameter
    global_params_reference_list = []
    for name, param_type in global_params_types.items():
        if param_type == "vector":
            global_params_reference_list.extend(global_params_reference_dict[name])
        elif param_type == "scalar":
            global_params_reference_list.append(global_params_reference_dict[name])
        else:
            raise ValueError(
                f"Global parameter '{name}' has unsupported type '{param_type}'. "
                f"Must be 'vector' or 'scalar'."
            )

    # Convert to numpy array and store in data container
    data.global_params_reference = np.array(
        global_params_reference_list, dtype=np.float32
    )

    return data

def process_global_params(
    data: ExternalAerodynamicsExtractedDataInMemory,
    global_parameters: dict,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Base processor for global parameters - to be overridden for specific datasets.

    This is a placeholder that should be replaced by dataset-specific implementations
    (e.g., process_global_params_hlpw).

    By default, sets global_params_values equal to global_params_reference,
    assuming simulation conditions match reference conditions.

    Args:
        data: Container with simulation data and metadata
        global_parameters: Dict from config with parameter definitions

    Returns:
        Updated `data` with global_params_values set
    """
    # Default behavior: assume simulation values match reference
    data.global_params_values = data.global_params_reference.copy()

    return data


# ============================================================================
# Case-Specific Processors
# ============================================================================
# These functions demonstrate how to extract global_params_values from
# simulation data for specific datasets. Replace process_global_params above
# with these in your config for case-specific processing.

WIND_DIRECTION_MAP = {
    "N": [0.0,  -2.0],   # North wind blows southward (negative V)
    "S": [0.0,   2.0],   # South wind blows northward (positive V)
    "E": [-2.0,  0.0],   # East wind blows westward (negative U)
    "W": [2.0,   0.0],   # West wind blows eastward (positive U)
}

def process_global_params_hdb(
    data: ExternalAerodynamicsExtractedDataInMemory,
    global_parameters: dict,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Extract global parameters from HDB wind simulation filename.

    Extracts wind direction from filename suffix (e.g. case_HDB_1531b136_S -> S)
    and maps it to the actual [windU, windV] inlet velocity vector.

    Args:
        data: Container with simulation data and metadata
        global_parameters: Dict from config with parameter definitions

    Returns:
        Updated data with global_params_values set from filename
    """
    filename = data.metadata.filename

    # Extract direction suffix — filename format: case_HDB_<hash>_<Direction>
    # e.g. case_HDB_1531b136_S -> "S"
    direction = filename.split("_")[-1].upper()

    if direction not in WIND_DIRECTION_MAP:
        raise ValueError(
            f"Could not extract wind direction from filename '{filename}'. "
            f"Expected suffix to be one of {list(WIND_DIRECTION_MAP.keys())}, "
            f"got '{direction}'."
        )

    wind_vector = WIND_DIRECTION_MAP[direction]
    logger.info(f"[{filename}] Wind direction={direction}, vector={wind_vector}")

    # Build flattened values array in same order as global_parameters dict
    global_params_values_list = []
    for name, params in global_parameters.items():
        param_type = params["type"]
        if name == "inlet_velocity":
            if param_type != "vector":
                raise ValueError("inlet_velocity must have type: vector")
            global_params_values_list.extend(wind_vector)
        elif name == "air_density":
            if param_type != "scalar":
                raise ValueError("air_density must have type: scalar")
            global_params_values_list.append(params["reference"])  # constant across cases
        else:
            raise ValueError(
                f"Unknown global parameter '{name}'. "
                f"Add handling for it in hdb_global_params_processor.py"
            )

    data.global_params_values = np.array(global_params_values_list, dtype=np.float32)
    logger.info(f"[{filename}] global_params_values={data.global_params_values}")

    return data