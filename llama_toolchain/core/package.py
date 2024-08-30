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
from pydantic import BaseModel

from termcolor import cprint

from llama_toolchain.common.config_dirs import BUILDS_BASE_DIR
from llama_toolchain.common.exec import run_with_pty
from llama_toolchain.common.serialize import EnumEncoder

from llama_toolchain.core.datatypes import *  # noqa: F403
from llama_toolchain.core.distribution import api_providers, SERVER_DEPENDENCIES


class BuildType(Enum):
    container = "container"
    conda_env = "conda_env"

    def descriptor(self) -> str:
        return "docker" if self == self.container else "conda"


class Dependencies(BaseModel):
    pip_packages: List[str]
    docker_image: Optional[str] = None


class ApiInput(BaseModel):
    api: Api
    provider: str


def build_package(
    api_inputs: List[ApiInput],
    build_type: BuildType,
    name: str,
    distribution_id: Optional[str] = None,
    docker_image: Optional[str] = None,
):
    if not distribution_id:
        distribution_id = "adhoc"

    build_dir = BUILDS_BASE_DIR / distribution_id / build_type.descriptor()
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

        stub_config[api.value] = {"provider_id": api_input.provider}

    if package_file.exists():
        cprint(
            f"Build `{package_name}` exists; will reconfigure",
            color="yellow",
        )
        c = PackageConfig(**yaml.safe_load(package_file.read_text()))
        for api_str, new_config in stub_config.items():
            if api_str not in c.providers:
                c.providers[api_str] = new_config
            else:
                existing_config = c.providers[api_str]
                if existing_config["provider_id"] != new_config["provider_id"]:
                    cprint(
                        f"Provider `{api_str}` has changed from `{existing_config}` to `{new_config}`",
                        color="yellow",
                    )
                    c.providers[api_str] = new_config
    else:
        c = PackageConfig(
            built_at=datetime.now(),
            package_name=package_name,
            providers=stub_config,
        )

    c.distribution_id = distribution_id
    c.docker_image = package_name if build_type == BuildType.container else None
    c.conda_env = package_name if build_type == BuildType.conda_env else None

    with open(package_file, "w") as f:
        to_write = json.loads(json.dumps(c.dict(), cls=EnumEncoder))
        f.write(yaml.dump(to_write, sort_keys=False))

    if build_type == BuildType.container:
        script = pkg_resources.resource_filename(
            "llama_toolchain", "core/build_container.sh"
        )
        args = [
            script,
            distribution_id,
            package_name,
            package_deps.docker_image,
            " ".join(package_deps.pip_packages),
        ]
    else:
        script = pkg_resources.resource_filename(
            "llama_toolchain", "core/build_conda_env.sh"
        )
        args = [
            script,
            distribution_id,
            package_name,
            " ".join(package_deps.pip_packages),
        ]

    return_code = run_with_pty(args)
    if return_code != 0:
        cprint(
            f"Failed to build target {package_name} with return code {return_code}",
            color="red",
        )
        return

    cprint(
        f"Target `{package_name}` built with configuration at {str(package_file)}",
        color="green",
    )
