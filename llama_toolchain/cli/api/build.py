# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json
import os
from pydantic import BaseModel
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import pkg_resources
import yaml

from termcolor import cprint

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.common.config_dirs import BUILDS_BASE_DIR
from llama_toolchain.distribution.datatypes import *  # noqa: F403


class BuildType(Enum):
    container = "container"
    conda_env = "conda_env"


class Dependencies(BaseModel):
    pip_packages: List[str]
    docker_image: Optional[str] = None


def get_dependencies(
    provider: ProviderSpec, dependencies: Dict[str, ProviderSpec]
) -> Dependencies:
    from llama_toolchain.distribution.distribution import SERVER_DEPENDENCIES

    pip_packages = provider.pip_packages
    for dep in dependencies.values():
        if dep.docker_image:
            raise ValueError(
                "You can only have the root provider specify a docker image"
            )
        pip_packages.extend(dep.pip_packages)

    return Dependencies(
        docker_image=provider.docker_image,
        pip_packages=pip_packages + SERVER_DEPENDENCIES
    )


def parse_dependencies(
    dependencies: str, parser: argparse.ArgumentParser
) -> Dict[str, ProviderSpec]:
    from llama_toolchain.distribution.distribution import api_providers

    all_providers = api_providers()

    deps = {}
    for dep in dependencies.split(","):
        dep = dep.strip()
        if not dep:
            continue
        api_str, provider = dep.split("=")
        api = Api(api_str)

        provider = provider.strip()
        if provider not in all_providers[api]:
            parser.error(f"Provider `{provider}` is not available for API `{api}`")
            return
        deps[api] = all_providers[api][provider]

    return deps


class ApiBuild(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "build",
            prog="llama api build",
            description="Build a Llama stack API provider container",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_api_build_command)

    def _add_arguments(self):
        from llama_toolchain.distribution.distribution import stack_apis

        allowed_args = [a.name for a in stack_apis()]
        self.parser.add_argument(
            "api",
            choices=allowed_args,
            help="Stack API (one of: {})".format(", ".join(allowed_args)),
        )

        self.parser.add_argument(
            "--provider",
            type=str,
            help="The provider to package into the container",
            required=True,
        )
        self.parser.add_argument(
            "--dependencies",
            type=str,
            help="Comma separated list of (downstream_api=provider) dependencies needed for the API",
            required=False,
        )
        self.parser.add_argument(
            "--name",
            type=str,
            help="Name of the build target (image, conda env)",
            required=True,
        )
        self.parser.add_argument(
            "--type",
            type=str,
            default="container",
            choices=[v.value for v in BuildType],
        )

    def _run_api_build_command(self, args: argparse.Namespace) -> None:
        from llama_toolchain.common.exec import run_with_pty
        from llama_toolchain.distribution.distribution import api_providers
        from llama_toolchain.common.serialize import EnumEncoder

        os.makedirs(BUILDS_BASE_DIR, exist_ok=True)
        all_providers = api_providers()

        api = Api(args.api)
        assert api in all_providers

        providers = all_providers[api]
        if args.provider not in providers:
            self.parser.error(
                f"Provider `{args.provider}` is not available for API `{api}`"
            )
            return

        if args.type == BuildType.container.value:
            package_name = f"image-{args.provider}-{args.name}"
        else:
            package_name = f"env-{args.provider}-{args.name}"
        package_name = package_name.replace("::", "-")

        build_dir = BUILDS_BASE_DIR / args.api
        os.makedirs(build_dir, exist_ok=True)

        # get these names straight. too confusing.
        provider_deps = parse_dependencies(args.dependencies or "", self.parser)
        dependencies = get_dependencies(providers[args.provider], provider_deps)

        package_file = build_dir / f"{package_name}.yaml"

        stub_config = {
            api.value: {
                "provider_id": args.provider,
            },
            **provider_deps,
        }
        with open(package_file, "w") as f:
            c = PackageConfig(
                built_at=datetime.now(),
                package_name=package_name,
                docker_image=(
                    package_name if args.type == BuildType.container.value else None
                ),
                conda_env=(
                    package_name if args.type == BuildType.conda_env.value else None
                ),
                providers=stub_config,
            )
            to_write = json.loads(json.dumps(c.dict(), cls=EnumEncoder))
            f.write(yaml.dump(to_write, sort_keys=False))

        if args.type == BuildType.container.value:
            script = pkg_resources.resource_filename(
                "llama_toolchain", "distribution/build_container.sh"
            )
            args = [
                script,
                args.api,
                package_name,
                dependencies.docker_image or "python:3.10-slim",
                " ".join(dependencies.pip_packages),
            ]
        else:
            script = pkg_resources.resource_filename(
                "llama_toolchain", "distribution/build_conda_env.sh"
            )
            args = [
                script,
                args.api,
                package_name,
                " ".join(dependencies.pip_packages),
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
