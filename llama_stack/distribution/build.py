# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import List, Optional

import pkg_resources
from pydantic import BaseModel

from termcolor import cprint

from llama_stack.distribution.utils.exec import run_with_pty

from llama_stack.distribution.datatypes import *  # noqa: F403
from pathlib import Path

from llama_stack.distribution.distribution import api_providers, SERVER_DEPENDENCIES


class ImageType(Enum):
    docker = "docker"
    conda = "conda"


class Dependencies(BaseModel):
    pip_packages: List[str]
    docker_image: Optional[str] = None


class ApiInput(BaseModel):
    api: Api
    provider: str


def build_image(build_config: BuildConfig, build_file_path: Path):
    package_deps = Dependencies(
        docker_image=build_config.distribution_spec.docker_image or "python:3.10-slim",
        pip_packages=SERVER_DEPENDENCIES,
    )

    # extend package dependencies based on providers spec
    all_providers = api_providers()
    for (
        api_str,
        provider_or_providers,
    ) in build_config.distribution_spec.providers.items():
        providers_for_api = all_providers[Api(api_str)]

        providers = (
            provider_or_providers
            if isinstance(provider_or_providers, list)
            else [provider_or_providers]
        )

        for provider in providers:
            if provider not in providers_for_api:
                raise ValueError(
                    f"Provider `{provider}` is not available for API `{api_str}`"
                )

            provider_spec = providers_for_api[provider]
            package_deps.pip_packages.extend(provider_spec.pip_packages)
            if provider_spec.docker_image:
                raise ValueError("A stack's dependencies cannot have a docker image")

    if build_config.image_type == ImageType.docker.value:
        script = pkg_resources.resource_filename(
            "llama_stack", "distribution/build_container.sh"
        )
        args = [
            script,
            build_config.name,
            package_deps.docker_image,
            str(build_file_path),
            " ".join(package_deps.pip_packages),
        ]
    else:
        script = pkg_resources.resource_filename(
            "llama_stack", "distribution/build_conda_env.sh"
        )
        args = [
            script,
            build_config.name,
            " ".join(package_deps.pip_packages),
        ]

    return_code = run_with_pty(args)
    if return_code != 0:
        cprint(
            f"Failed to build target {build_config.name} with return code {return_code}",
            color="red",
        )
        return
