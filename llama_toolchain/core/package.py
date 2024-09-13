# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os
from datetime import datetime
from enum import Enum
from typing import List, Optional

import pkg_resources
import yaml

from llama_toolchain.common.config_dirs import BUILDS_BASE_DIR
from llama_toolchain.common.exec import run_with_pty
from llama_toolchain.common.serialize import EnumEncoder
from pydantic import BaseModel

from termcolor import cprint

from llama_toolchain.core.datatypes import *  # noqa: F403
from llama_toolchain.core.distribution import api_providers, SERVER_DEPENDENCIES


class ImageType(Enum):
    docker = "docker"
    conda = "conda"


class Dependencies(BaseModel):
    pip_packages: List[str]
    docker_image: Optional[str] = None


class ApiInput(BaseModel):
    api: Api
    provider: str


def build_package(
    api_inputs: List[ApiInput],
    image_type: ImageType,
    name: str,
    distribution_type: Optional[str] = None,
    docker_image: Optional[str] = None,
):
    if not distribution_type:
        distribution_type = "adhoc"

    build_dir = BUILDS_BASE_DIR / distribution_type / image_type.value
    os.makedirs(build_dir, exist_ok=True)

    package_name = name.replace("::", "-")
    package_file = build_dir / f"{package_name}.yaml"

    all_providers = api_providers()

    package_deps = Dependencies(
        docker_image=docker_image or "python:3.10-slim",
        pip_packages=SERVER_DEPENDENCIES,
    )

    stub_config = {}
    for api_input in api_inputs:
        api = api_input.api
        providers_for_api = all_providers[api]
        if api_input.provider not in providers_for_api:
            raise ValueError(
                f"Provider `{api_input.provider}` is not available for API `{api}`"
            )

        provider = providers_for_api[api_input.provider]
        package_deps.pip_packages.extend(provider.pip_packages)
        if provider.docker_image:
            raise ValueError("A stack's dependencies cannot have a docker image")

        stub_config[api.value] = {"provider_type": api_input.provider}

    if image_type == ImageType.docker:
        script = pkg_resources.resource_filename(
            "llama_toolchain", "core/build_container.sh"
        )
        args = [
            script,
            distribution_type,
            package_name,
            package_deps.docker_image,
            str(package_file),
            " ".join(package_deps.pip_packages),
        ]
    else:
        script = pkg_resources.resource_filename(
            "llama_toolchain", "core/build_conda_env.sh"
        )
        args = [
            script,
            distribution_type,
            package_name,
            str(package_file),
            " ".join(package_deps.pip_packages),
        ]

    return_code = run_with_pty(args)
    if return_code != 0:
        cprint(
            f"Failed to build target {package_name} with return code {return_code}",
            color="red",
        )
        return
