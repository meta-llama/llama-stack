# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from typing import Any

import psutil
import torch


def setup_environment():
    """Setup common environment variables for training."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "0"
    os.environ["MKL_NUM_THREADS"] = "1"


def get_gb(to_convert: int) -> str:
    """Converts memory stats to GB and formats to 2 decimal places.
    Args:
        to_convert: Memory value in bytes
    Returns:
        str: Memory value in GB formatted to 2 decimal places
    """
    return f"{(to_convert / (1024**3)):.2f}"


def get_memory_stats(device: torch.device) -> dict[str, Any]:
    """Get memory statistics for the given device."""
    stats = {
        "system_memory": {
            "total": get_gb(psutil.virtual_memory().total),
            "available": get_gb(psutil.virtual_memory().available),
            "used": get_gb(psutil.virtual_memory().used),
            "percent": psutil.virtual_memory().percent,
        }
    }

    if device.type == "cuda":
        stats["device_memory"] = {
            "allocated": get_gb(torch.cuda.memory_allocated(device)),
            "reserved": get_gb(torch.cuda.memory_reserved(device)),
            "max_allocated": get_gb(torch.cuda.max_memory_allocated(device)),
        }
    elif device.type == "mps":
        # MPS doesn't provide direct memory stats, but we can track system memory
        stats["device_memory"] = {
            "note": "MPS memory stats not directly available",
            "system_memory_used": get_gb(psutil.virtual_memory().used),
        }
    elif device.type == "cpu":
        # For CPU, we track process memory usage
        process = psutil.Process()
        stats["device_memory"] = {
            "process_rss": get_gb(process.memory_info().rss),
            "process_vms": get_gb(process.memory_info().vms),
            "process_percent": process.memory_percent(),
        }

    return stats


def setup_torch_device(device_str: str) -> torch.device:
    """Initialize and validate a PyTorch device.
    This function handles device initialization and validation for different device types:
    - CUDA: Validates CUDA availability and handles device selection
    - MPS: Validates MPS availability for Apple Silicon
    - CPU: Basic validation
    - HPU: Raises error as it's not supported
    Args:
        device_str: String specifying the device ('cuda', 'cpu', 'mps')
    Returns:
        torch.device: The initialized and validated device
    Raises:
        RuntimeError: If device initialization fails or device is not supported
    """
    try:
        device = torch.device(device_str)
    except RuntimeError as e:
        raise RuntimeError(f"Error getting Torch Device {str(e)}") from e

    # Validate device capabilities
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{device.type}: Torch has no CUDA/ROCm support or could not detect a compatible device."
            )
        if device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(f"{device.type}: Torch has no MPS support or could not detect a compatible device.")
    elif device.type == "hpu":
        raise RuntimeError(f"{device.type}: training does not support Intel Gaudi.")

    return device


async def setup_data(datasetio_api, dataset_id: str) -> list[dict[str, Any]]:
    """Load dataset from llama stack dataset provider"""
    try:
        all_rows = await datasetio_api.iterrows(
            dataset_id=dataset_id,
            limit=-1,
        )
        if not isinstance(all_rows.data, list):
            raise RuntimeError("Expected dataset data to be a list")
        return all_rows.data
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {str(e)}") from e
