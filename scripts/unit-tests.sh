#!/bin/sh

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree. 

uv run -p 3.10.16 --with . --with ".[dev]" --with ".[test]" pytest -s -v tests/unit/ --junitxml=pytest-report.xml
