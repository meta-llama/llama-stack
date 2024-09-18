# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_stack.cli.subcommand import Subcommand
from llama_stack.distribution.datatypes import *  # noqa: F403
from pathlib import Path

import yaml
from prompt_toolkit.validation import Validator
from termcolor import cprint


class StackBuild(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "build",
            prog="llama stack build",
            description="Build a Llama stack container",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_stack_build_command)

    def _add_arguments(self):
        self.parser.add_argument(
            "config",
            type=str,
            default=None,
            nargs="*",
            help="Path to a config file to use for the build. You can find example configs in llama_stack/distribution/example_configs. If this argument is not provided, you will be prompted to enter information interactively",
        )

        self.parser.add_argument(
            "--name",
            type=str,
            help="Name of the llama stack build to override from template config",
        )

    def _run_stack_build_command_from_build_config(
        self, build_config: BuildConfig
    ) -> None:
        import json
        import os

        from llama_stack.distribution.build import ApiInput, build_image, ImageType

        from llama_stack.distribution.utils.config_dirs import DISTRIBS_BASE_DIR
        from llama_stack.distribution.utils.serialize import EnumEncoder

        # save build.yaml spec for building same distribution again
        if build_config.image_type == ImageType.docker.value:
            # docker needs build file to be in the llama-stack repo dir to be able to copy over to the image
            llama_stack_path = Path(os.path.relpath(__file__)).parent.parent.parent
            build_dir = (
                llama_stack_path / "configs/distributions" / build_config.image_type
            )
        else:
            build_dir = (
                Path(os.getenv("CONDA_PREFIX")).parent
                / f"llamastack-{build_config.name}"
            )

        os.makedirs(build_dir, exist_ok=True)
        build_file_path = build_dir / f"{build_config.name}-build.yaml"

        with open(build_file_path, "w") as f:
            to_write = json.loads(json.dumps(build_config.dict(), cls=EnumEncoder))
            f.write(yaml.dump(to_write, sort_keys=False))

        build_image(build_config, build_file_path)

        cprint(
            f"Build spec configuration saved at {str(build_file_path)}",
            color="green",
        )

    def _run_stack_build_command(self, args: argparse.Namespace) -> None:
        from llama_stack.distribution.distribution import Api, api_providers
        from llama_stack.distribution.utils.dynamic import instantiate_class_type
        from prompt_toolkit import prompt

        if not args.config:
            name = prompt(
                "> Enter an unique name for identifying your Llama Stack build distribution (e.g. my-local-stack): "
            )
            image_type = prompt(
                "> Enter the image type you want your distribution to be built with (docker or conda): ",
                validator=Validator.from_callable(
                    lambda x: x in ["docker", "conda"],
                    error_message="Invalid image type, please enter (conda|docker)",
                ),
                default="conda",
            )

            cprint(
                f"\n Llama Stack is composed of several APIs working together. Let's configure the providers (implementations) you want to use for these APIs.",
                color="green",
            )

            providers = dict()
            for api in Api:
                all_providers = api_providers()
                providers_for_api = all_providers[api]

                api_provider = prompt(
                    "> Enter the API provider for the {} API: (default=meta-reference): ".format(
                        api.value
                    ),
                    validator=Validator.from_callable(
                        lambda x: x in providers_for_api,
                        error_message="Invalid provider, please enter one of the following: {}".format(
                            providers_for_api.keys()
                        ),
                    ),
                    default=(
                        "meta-reference"
                        if "meta-reference" in providers_for_api
                        else list(providers_for_api.keys())[0]
                    ),
                )

                providers[api.value] = api_provider

            description = prompt(
                "\n > (Optional) Enter a short description for your Llama Stack distribution: ",
                default="",
            )

            distribution_spec = DistributionSpec(
                providers=providers,
                description=description,
            )

            build_config = BuildConfig(
                name=name, image_type=image_type, distribution_spec=distribution_spec
            )
            self._run_stack_build_command_from_build_config(build_config)
            return

        with open(args.config, "r") as f:
            try:
                build_config = BuildConfig(**yaml.safe_load(f))
            except Exception as e:
                self.parser.error(f"Could not parse config file {args.config}: {e}")
                return
            if args.name:
                build_config.name = args.name
            self._run_stack_build_command_from_build_config(build_config)
