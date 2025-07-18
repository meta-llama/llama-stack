# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from pathlib import Path
from typing import Optional


def get_xdg_config_home() -> Path:
    """
    Get the XDG config home directory.
    
    Returns:
        Path: XDG_CONFIG_HOME if set, otherwise ~/.config
    """
    return Path(os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")))


def get_xdg_data_home() -> Path:
    """
    Get the XDG data home directory.
    
    Returns:
        Path: XDG_DATA_HOME if set, otherwise ~/.local/share
    """
    return Path(os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")))


def get_xdg_cache_home() -> Path:
    """
    Get the XDG cache home directory.
    
    Returns:
        Path: XDG_CACHE_HOME if set, otherwise ~/.cache
    """
    return Path(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")))


def get_xdg_state_home() -> Path:
    """
    Get the XDG state home directory.
    
    Returns:
        Path: XDG_STATE_HOME if set, otherwise ~/.local/state
    """
    return Path(os.environ.get("XDG_STATE_HOME", os.path.expanduser("~/.local/state")))


def get_llama_stack_config_dir() -> Path:
    """
    Get the llama-stack configuration directory.
    
    This function provides backwards compatibility by checking for the legacy
    LLAMA_STACK_CONFIG_DIR environment variable first, then falling back to
    XDG-compliant paths.
    
    Returns:
        Path: Configuration directory for llama-stack
    """
    # Check for legacy environment variable first for backwards compatibility
    legacy_dir = os.environ.get("LLAMA_STACK_CONFIG_DIR")
    if legacy_dir:
        return Path(legacy_dir)
    
    # Check if legacy ~/.llama directory exists and contains data
    legacy_path = Path.home() / ".llama"
    if legacy_path.exists() and any(legacy_path.iterdir()):
        return legacy_path
    
    # Use XDG-compliant path
    return get_xdg_config_home() / "llama-stack"


def get_llama_stack_data_dir() -> Path:
    """
    Get the llama-stack data directory.
    
    This is used for persistent data like model checkpoints.
    
    Returns:
        Path: Data directory for llama-stack
    """
    # Check for legacy environment variable first for backwards compatibility
    legacy_dir = os.environ.get("LLAMA_STACK_CONFIG_DIR")
    if legacy_dir:
        return Path(legacy_dir)
    
    # Check if legacy ~/.llama directory exists and contains data
    legacy_path = Path.home() / ".llama"
    if legacy_path.exists() and any(legacy_path.iterdir()):
        return legacy_path
    
    # Use XDG-compliant path
    return get_xdg_data_home() / "llama-stack"


def get_llama_stack_cache_dir() -> Path:
    """
    Get the llama-stack cache directory.
    
    This is used for temporary/cache data.
    
    Returns:
        Path: Cache directory for llama-stack
    """
    # Check for legacy environment variable first for backwards compatibility
    legacy_dir = os.environ.get("LLAMA_STACK_CONFIG_DIR")
    if legacy_dir:
        return Path(legacy_dir)
    
    # Check if legacy ~/.llama directory exists and contains data
    legacy_path = Path.home() / ".llama"
    if legacy_path.exists() and any(legacy_path.iterdir()):
        return legacy_path
    
    # Use XDG-compliant path
    return get_xdg_cache_home() / "llama-stack"


def get_llama_stack_state_dir() -> Path:
    """
    Get the llama-stack state directory.
    
    This is used for runtime state data.
    
    Returns:
        Path: State directory for llama-stack
    """
    # Check for legacy environment variable first for backwards compatibility
    legacy_dir = os.environ.get("LLAMA_STACK_CONFIG_DIR")
    if legacy_dir:
        return Path(legacy_dir)
    
    # Check if legacy ~/.llama directory exists and contains data
    legacy_path = Path.home() / ".llama"
    if legacy_path.exists() and any(legacy_path.iterdir()):
        return legacy_path
    
    # Use XDG-compliant path
    return get_xdg_state_home() / "llama-stack"


def get_xdg_compliant_path(
    path_type: str, 
    subdirectory: Optional[str] = None,
    legacy_fallback: bool = True
) -> Path:
    """
    Get an XDG-compliant path for a given type.
    
    Args:
        path_type: Type of path ('config', 'data', 'cache', 'state')
        subdirectory: Optional subdirectory within the base path
        legacy_fallback: Whether to check for legacy ~/.llama directory
        
    Returns:
        Path: XDG-compliant path
        
    Raises:
        ValueError: If path_type is not recognized
    """
    path_map = {
        "config": get_llama_stack_config_dir,
        "data": get_llama_stack_data_dir,
        "cache": get_llama_stack_cache_dir,
        "state": get_llama_stack_state_dir,
    }
    
    if path_type not in path_map:
        raise ValueError(f"Unknown path type: {path_type}. Must be one of: {list(path_map.keys())}")
    
    base_path = path_map[path_type]()
    
    if subdirectory:
        return base_path / subdirectory
    
    return base_path


def migrate_legacy_directory() -> bool:
    """
    Migrate from legacy ~/.llama directory to XDG-compliant directories.
    
    This function helps users migrate their existing data to the new
    XDG-compliant structure.
    
    Returns:
        bool: True if migration was successful or not needed, False otherwise
    """
    legacy_path = Path.home() / ".llama"
    
    if not legacy_path.exists():
        return True  # No migration needed
    
    print(f"Found legacy directory at {legacy_path}")
    print("Consider migrating to XDG-compliant directories:")
    print(f"  Config: {get_llama_stack_config_dir()}")
    print(f"  Data: {get_llama_stack_data_dir()}")
    print(f"  Cache: {get_llama_stack_cache_dir()}")
    print(f"  State: {get_llama_stack_state_dir()}")
    print("Migration can be done by moving the appropriate subdirectories.")
    
    return True


def ensure_directory_exists(path: Path) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory
    """
    path.mkdir(parents=True, exist_ok=True) 