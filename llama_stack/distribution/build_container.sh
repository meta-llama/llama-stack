#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Function to get Python command
get_python_cmd() {
    if is_command_available python; then
        echo "python"
    elif is_command_available python3; then
        echo "python3"
    else
        echo "Error: Neither python nor python3 is installed. Please install Python to continue." >&2
        exit 1
    fi
}
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common.sh"

# call the python script
CMD=$(get_python_cmd)
$CMD $SCRIPT_DIR/build_container.py "$@"
