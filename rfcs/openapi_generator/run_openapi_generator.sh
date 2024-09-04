#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

PYTHONPATH=${PYTHONPATH:-}

set -euo pipefail

missing_packages=()

check_package() {
    if ! pip show "$1" &>/dev/null; then
        missing_packages+=("$1")
    fi
}

check_package json-strong-typing

if [ ${#missing_packages[@]} -ne 0 ]; then
    echo "Error: The following package(s) are not installed:"
    printf " - %s\n" "${missing_packages[@]}"
    echo "Please install them using:"
    echo "pip install ${missing_packages[*]}"
    exit 1
fi

PYTHONPATH=$PYTHONPATH:../.. python3 -m rfcs.openapi_generator.generate $*
