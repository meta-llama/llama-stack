# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import importlib.resources
import logging
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel
from termcolor import cprint

from llama_stack.distribution.datatypes import BuildConfig, Provider
from llama_stack.distribution.distribution import get_provider_registry
from llama_stack.distribution.utils.exec import run_command
from llama_stack.distribution.utils.image_types import LlamaStackImageType
from llama_stack.providers.datatypes import Api

log = logging.getLogger(__name__)

# These are the dependencies needed by the distribution server.
# `llama-stack` is automatically installed by the installation script.
SERVER_DEPENDENCIES = [
    "aiosqlite",
    "fastapi",
    "fire",
    "httpx",
    "uvicorn",
]


class ApiInput(BaseModel):
    api: Api
    provider: str


def get_provider_dependencies(
    config_providers: Dict[str, List[Provider]],
) -> tuple[list[str], list[str]]:
    """Get normal and special dependencies from provider configuration."""
    all_providers = get_provider_registry()
    deps = []

    for api_str, provider_or_providers in config_providers.items():
        providers_for_api = all_providers[Api(api_str)]

        providers = provider_or_providers if isinstance(provider_or_providers, list) else [provider_or_providers]

        for provider in providers:
            # Providers from BuildConfig and RunConfig are subtly different – not great
            provider_type = provider if isinstance(provider, str) else provider.provider_type

            if provider_type not in providers_for_api:
                raise ValueError(f"Provider `{provider}` is not available for API `{api_str}`")

            provider_spec = providers_for_api[provider_type]
            deps.extend(provider_spec.pip_packages)
            if provider_spec.container_image:
                raise ValueError("A stack's dependencies cannot have a container image")

    normal_deps = []
    special_deps = []
    for package in deps:
        if "--no-deps" in package or "--index-url" in package:
            special_deps.append(package)
        else:
            normal_deps.append(package)

    return list(set(normal_deps)), list(set(special_deps))


def print_pip_install_help(providers: Dict[str, List[Provider]]):
    normal_deps, special_deps = get_provider_dependencies(providers)

    cprint(
        f"Please install needed dependencies using the following commands:\n\nuv pip install {' '.join(normal_deps)}",
        "yellow",
    )
    for special_dep in special_deps:
        cprint(f"uv pip install {special_dep}", "yellow")
    print()


def build_image(
    build_config: BuildConfig,
    build_file_path: Path,
    image_name: str,
    template_or_config: str,
):
    container_base = build_config.distribution_spec.container_image or "python:3.10-slim"

    normal_deps, special_deps = get_provider_dependencies(build_config.distribution_spec.providers)
    normal_deps += SERVER_DEPENDENCIES

    if build_config.image_type == LlamaStackImageType.CONTAINER.value:
        script = str(importlib.resources.files("llama_stack") / "distribution/build_container.sh")
        args = [
            script,
            template_or_config,
            image_name,
            container_base,
            " ".join(normal_deps),
        ]
    elif build_config.image_type == LlamaStackImageType.CONDA.value:
        script = str(importlib.resources.files("llama_stack") / "distribution/build_conda_env.sh")
        args = [
            script,
            str(image_name),
            str(build_file_path),
            " ".join(normal_deps),
        ]
    elif build_config.image_type == LlamaStackImageType.VENV.value:
        script = str(importlib.resources.files("llama_stack") / "distribution/build_venv.sh")
        args = [
            script,
            str(image_name),
            " ".join(normal_deps),
        ]

    if special_deps:
        args.append("#".join(special_deps))

    return_code = run_command(args)

    if return_code != 0:
        log.error(
            f"Failed to build target {image_name} with return code {return_code}",
        )

    return return_code
