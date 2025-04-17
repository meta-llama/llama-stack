# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration  # filtered out of the fast suite


def _tmp_yaml() -> Path:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    yaml.safe_dump(
        {
            "version": "2",
            "distribution_spec": {
                "description": "UBI9 compile smoke",
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


def test_image_compiles_c():
    cfg = _tmp_yaml()
    tag = "ci-test:dev"

    # Build image (providers‑build.yml already did `uv venv` etc.)
    subprocess.run(
        ["llama", "stack", "build", "--config", cfg, "--image-name", "ci-test"],
        check=True,
        env={**os.environ, "USE_COPY_NOT_MOUNT": "true"},
    )

    # compile a hello‑world c program inside the container
    hello_c = textwrap.dedent("""
        #include <stdio.h>
        int main(){puts("ok");return 0;}
    """)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".c") as f:
        f.write(hello_c.encode())

    subprocess.run(
        ["docker", "run", "--rm", "-v", f.name + ":/tmp/a.c", tag, "bash", "-c", "gcc /tmp/a.c -o /tmp/a && /tmp/a"],
        check=True,
    )
