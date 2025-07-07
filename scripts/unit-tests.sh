#!/bin/sh

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

PYTHON_VERSION=${PYTHON_VERSION:-3.12}

command -v uv >/dev/null 2>&1 || { echo >&2 "uv is required but it's not installed. Exiting."; exit 1; }

uv python find "$PYTHON_VERSION"
FOUND_PYTHON=$?
if [ $FOUND_PYTHON -ne 0 ]; then
     uv python install "$PYTHON_VERSION"
fi

# Install coverage if not installed
uv pip install coverage >/dev/null 2>&1

uv run --python "$PYTHON_VERSION" --with-editable . --group unit \
    coverage run --source=llama_stack -m pytest --asyncio-mode=auto -s -v tests/unit/ "$@" && \
uv run --python "$PYTHON_VERSION" coverage html -d htmlcov-$PYTHON_VERSION

