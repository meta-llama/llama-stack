# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from pathlib import Path

from .xdg_utils import (
    get_llama_stack_config_dir,
    get_llama_stack_data_dir,
    get_llama_stack_state_dir,
    get_xdg_compliant_path,
)

# Base directory for all llama-stack configuration
# This now uses XDG-compliant paths with backwards compatibility
LLAMA_STACK_CONFIG_DIR = get_llama_stack_config_dir()

# Distribution configurations - stored in config directory
DISTRIBS_BASE_DIR = LLAMA_STACK_CONFIG_DIR / "distributions"

# Model checkpoints - stored in data directory (persistent data)
DEFAULT_CHECKPOINT_DIR = get_llama_stack_data_dir() / "checkpoints"

# Runtime data - stored in state directory
RUNTIME_BASE_DIR = get_llama_stack_state_dir() / "runtime"

# External providers - stored in config directory
EXTERNAL_PROVIDERS_DIR = LLAMA_STACK_CONFIG_DIR / "providers.d"

# Legacy compatibility: if the legacy environment variable is set, use it for all paths
# This ensures that existing installations continue to work
legacy_config_dir = os.getenv("LLAMA_STACK_CONFIG_DIR")
if legacy_config_dir:
    legacy_base = Path(legacy_config_dir)
    LLAMA_STACK_CONFIG_DIR = legacy_base
    DISTRIBS_BASE_DIR = legacy_base / "distributions"
    DEFAULT_CHECKPOINT_DIR = legacy_base / "checkpoints"
    RUNTIME_BASE_DIR = legacy_base / "runtime"
    EXTERNAL_PROVIDERS_DIR = legacy_base / "providers.d"
