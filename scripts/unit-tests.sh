#!/bin/sh

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

PYTHON_VERSION=${PYTHON_VERSION:-3.10}

command -v uv >/dev/null 2>&1 || { echo >&2 "uv is required but it's not installed. Exiting."; exit 1; }

uv python find $PYTHON_VERSION
FOUND_PYTHON=$?
if [ $FOUND_PYTHON -ne 0 ]; then
     uv python install $PYTHON_VERSION
fi

uv run --python $PYTHON_VERSION --with-editable . --with-editable ".[dev]" --with-editable ".[unit]" pytest --asyncio-mode=auto -s -v tests/unit/ $@
