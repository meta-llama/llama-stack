# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# tests/integration/test_ubi9_toolchain.py

from pathlib import Path


def test_ubi9_compiler_packages_present():
    """
    Verify that the UBI9 dnf install line in build_container.sh includes
    python3.11‑setuptools, python3.11‑devel, gcc, and make.
    """
    script = (
        Path(__file__).parents[2]  # moves from tests/integration up to repo root
        / "llama_stack"
        / "distribution"
        / "build_container.sh"
    )
    content = script.read_text(encoding="utf-8")

    expected = "python3.11-setuptools python3.11-devel gcc make"
    assert expected in content, (
        f"Expected to find '{expected}' in the UBI9 install line, but it was missing.\n\n"
        f"Content of {script}:\n{content}"
    )
