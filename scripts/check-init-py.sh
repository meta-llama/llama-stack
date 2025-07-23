#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
#
# Check for missing __init__.py files in Python packages
# This script finds directories that contain Python files but are missing __init__.py

set -euo pipefail

PACKAGE_DIR="${1:-llama_stack}"

if [ ! -d "$PACKAGE_DIR" ]; then
    echo "ERROR: Package directory '$PACKAGE_DIR' does not exist"
    exit 1
fi

# Get all directories with Python files (excluding __init__.py)
# Use temp file approach for maximum compatibility (works with any POSIX shell)
py_dirs_temp=$(mktemp)
trap 'rm -f "$py_dirs_temp"' EXIT

find "$PACKAGE_DIR" \
    -type f \
    -name "*.py" ! -name "__init__.py" \
    ! -path "*/.venv/*" \
    ! -path "*/node_modules/*" \
    -exec dirname {} \; | sort -u > "$py_dirs_temp"

missing_init_files=0

while IFS= read -r dir; do
    if [ -n "$dir" ] && [ ! -f "$dir/__init__.py" ]; then
        echo "ERROR: Missing __init__.py in directory: $dir"
        echo "This directory contains Python files but no __init__.py, which may cause packaging issues."
        missing_init_files=1
    fi
done < "$py_dirs_temp"

exit $missing_init_files
