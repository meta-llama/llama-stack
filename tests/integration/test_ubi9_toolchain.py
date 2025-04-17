# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re
import subprocess
import tempfile
from pathlib import Path

import yaml

# exact packages we just added
REQUIRED_PKGS = ("python3.11-devel", "gcc", "make")


def _tmp_yaml() -> Path:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    yaml.safe_dump(
        {
            "version": "2",
            "distribution_spec": {
                "description": "CI smoke test",
                "providers": {
                    "inference": ["remote::ollama"],
                    "vector_io": ["inline::faiss"],
                },
                "container_image": "registry.access.redhat.com/ubi9",
            },
            "image_type": "container",
            "image_name": "ci-test",
        },
        f,
    )
    f.close()
    return Path(f.name)


def test_ubi9_toolchain_present():
    cfg = _tmp_yaml()

    # --dry-run only renders the Containerfile
    out = subprocess.run(
        ["llama", "stack", "build", "--config", cfg, "--dry-run"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    cfile = Path(re.search(r"(/tmp/.+?/Containerfile)", out).group(1)).read_text()
    missing = [p for p in REQUIRED_PKGS if p not in cfile]
    assert not missing, f"dnf line lost packages: {', '.join(missing)}"
