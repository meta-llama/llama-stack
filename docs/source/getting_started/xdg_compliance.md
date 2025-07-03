# XDG Base Directory Specification Compliance

Starting with version 0.2.14, Llama Stack follows the [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html) for organizing configuration and data files. This provides better integration with modern desktop environments and allows for more flexible customization of where files are stored.

## Overview

The XDG Base Directory Specification defines standard locations for different types of application data:

- **Configuration files** (`XDG_CONFIG_HOME`): User-specific configuration files
- **Data files** (`XDG_DATA_HOME`): User-specific data files that should persist
- **Cache files** (`XDG_CACHE_HOME`): User-specific cache files
- **State files** (`XDG_STATE_HOME`): User-specific state files

## Directory Mapping

Llama Stack now uses the following XDG-compliant directory structure:

| Data Type | XDG Directory | Default Location | Description |
|-----------|---------------|------------------|-------------|
| Configuration | `XDG_CONFIG_HOME` | `~/.config/llama-stack` | Distribution configs, provider configs |
| Data | `XDG_DATA_HOME` | `~/.local/share/llama-stack` | Model checkpoints, persistent files |
| Cache | `XDG_CACHE_HOME` | `~/.cache/llama-stack` | Temporary cache files |
| State | `XDG_STATE_HOME` | `~/.local/state/llama-stack` | Runtime state, databases |

## Environment Variables

You can customize the locations by setting these environment variables:

```bash
# Override the base directories
export XDG_CONFIG_HOME="/custom/config/path"
export XDG_DATA_HOME="/custom/data/path"
export XDG_CACHE_HOME="/custom/cache/path"
export XDG_STATE_HOME="/custom/state/path"

# Or override specific Llama Stack directories
export SQLITE_STORE_DIR="/custom/database/path"
export FILES_STORAGE_DIR="/custom/files/path"
```

## Backwards Compatibility

Llama Stack maintains full backwards compatibility with existing installations:

1. **Legacy Environment Variable**: If `LLAMA_STACK_CONFIG_DIR` is set, it will be used for all directories
2. **Legacy Directory Detection**: If `~/.llama` exists and contains data, it will continue to be used
3. **Gradual Migration**: New installations use XDG paths, existing installations continue to work

## Migration Guide

### Automatic Migration

Use the built-in migration command to move from legacy `~/.llama` to XDG-compliant directories:

```bash
# Preview what would be migrated
llama migrate-xdg --dry-run

# Perform the migration
llama migrate-xdg
```

### Manual Migration

If you prefer to migrate manually, here's the mapping:

```bash
# Create XDG directories
mkdir -p ~/.config/llama-stack
mkdir -p ~/.local/share/llama-stack
mkdir -p ~/.local/state/llama-stack

# Move configuration files
mv ~/.llama/distributions ~/.config/llama-stack/
mv ~/.llama/providers.d ~/.config/llama-stack/

# Move data files
mv ~/.llama/checkpoints ~/.local/share/llama-stack/

# Move state files
mv ~/.llama/runtime ~/.local/state/llama-stack/

# Clean up empty legacy directory
rmdir ~/.llama
```

### Environment Variables in Configurations

Template configurations now use XDG-compliant environment variables:

```yaml
# Old format
db_path: ${env.SQLITE_STORE_DIR:=~/.llama/distributions/ollama}/registry.db

# New format
db_path: ${env.SQLITE_STORE_DIR:=${env.XDG_STATE_HOME:-~/.local/state}/llama-stack/distributions/ollama}/registry.db
```

## Configuration Examples

### Using Custom XDG Directories

```bash
# Set custom XDG directories
export XDG_CONFIG_HOME="/opt/llama-stack/config"
export XDG_DATA_HOME="/opt/llama-stack/data"
export XDG_STATE_HOME="/opt/llama-stack/state"

# Start Llama Stack
llama stack run my-distribution.yaml
```

### Using Legacy Directory

```bash
# Continue using legacy directory
export LLAMA_STACK_CONFIG_DIR="/home/user/.llama"

# Start Llama Stack
llama stack run my-distribution.yaml
```

### Custom Database and File Locations

```bash
# Override specific directories
export SQLITE_STORE_DIR="/fast/ssd/llama-stack/databases"
export FILES_STORAGE_DIR="/large/disk/llama-stack/files"

# Start Llama Stack
llama stack run my-distribution.yaml
```

## Benefits of XDG Compliance

1. **Standards Compliance**: Follows established Linux/Unix conventions
2. **Better Organization**: Separates configuration, data, cache, and state files
3. **Flexibility**: Easy to customize storage locations
4. **Backup-Friendly**: Easier to backup just data files or just configuration
5. **Multi-User Support**: Better support for shared systems
6. **Cross-Platform**: Works consistently across different environments

## Template Updates

All distribution templates have been updated to use XDG-compliant paths:

- Database files use `XDG_STATE_HOME`
- Model checkpoints use `XDG_DATA_HOME`
- Configuration files use `XDG_CONFIG_HOME`
- Cache files use `XDG_CACHE_HOME`

## Troubleshooting

### Migration Issues

If you encounter issues during migration:

1. **Check Permissions**: Ensure you have write permissions to target directories
2. **Disk Space**: Verify sufficient disk space in target locations
3. **Existing Files**: Handle conflicts with existing files in target locations

### Environment Variable Conflicts

If you have multiple environment variables set:

1. `LLAMA_STACK_CONFIG_DIR` takes highest precedence
2. Individual `XDG_*` variables override defaults
3. Fallback to legacy `~/.llama` if it exists
4. Default to XDG standard paths for new installations

### Debugging Path Resolution

To see which paths Llama Stack is using:

```python
from llama_stack.distribution.utils.xdg_utils import (
    get_llama_stack_config_dir,
    get_llama_stack_data_dir,
    get_llama_stack_state_dir,
)

print(f"Config: {get_llama_stack_config_dir()}")
print(f"Data: {get_llama_stack_data_dir()}")
print(f"State: {get_llama_stack_state_dir()}")
```

## Future Considerations

- Container deployments will continue to use `/app` or similar paths
- Cloud deployments may use provider-specific storage systems
- The XDG specification primarily applies to local development and single-user systems 