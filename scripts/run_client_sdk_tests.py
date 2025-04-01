#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os
from pathlib import Path

import pytest

"""
Script for running api on AsyncLlamaStackAsLibraryClient with templates

Assuming directory structure:
- llama-stack
    - scripts
    - tests
        - api

Example command:

cd llama-stack
EXPORT TOGETHER_API_KEY=<..>
EXPORT FIREWORKS_API_KEY=<..>
./scripts/run_client_sdk_tests.py --templates together fireworks --report
"""

REPO_ROOT = Path(__file__).parent.parent
CLIENT_SDK_TESTS_RELATIVE_PATH = "tests/api/"


def main(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    templates_dir = REPO_ROOT / "llama_stack" / "templates"
    user_specified_templates = [templates_dir / t for t in args.templates] if args.templates else []
    for d in templates_dir.iterdir():
        if d.is_dir() and d.name != "__pycache__":
            template_configs = list(d.rglob("run.yaml"))
            if len(template_configs) == 0:
                continue
            config = template_configs[0]
            if user_specified_templates:
                if not any(config.parent == t for t in user_specified_templates):
                    continue
            os.environ["LLAMA_STACK_CONFIG"] = str(config)
            pytest_args = "--report" if args.report else ""
            pytest.main(
                [
                    pytest_args,
                    "-s",
                    "-v",
                    str(REPO_ROOT / CLIENT_SDK_TESTS_RELATIVE_PATH),
                ]
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="llama_test",
    )
    parser.add_argument("--templates", nargs="+")
    parser.add_argument("--report", action="store_true")
    main(parser)
