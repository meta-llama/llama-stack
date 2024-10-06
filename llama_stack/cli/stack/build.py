# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_stack.cli.subcommand import Subcommand
from llama_stack.distribution.datatypes import *  # noqa: F403
import os
from functools import lru_cache
from pathlib import Path

TEMPLATES_PATH = (
    Path(os.path.relpath(__file__)).parent.parent.parent / "distribution" / "templates"
)


@lru_cache()
def available_templates_specs() -> List[BuildConfig]:
    import yaml

    template_specs = []
    for p in TEMPLATES_PATH.rglob("*.yaml"):
        with open(p, "r") as f:
            build_config = BuildConfig(**yaml.safe_load(f))
            template_specs.append(build_config)

    return template_specs


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
            "--config",
            type=str,
            default=None,
            help="Path to a config file to use for the build. You can find example configs in llama_stack/distribution/example_configs. If this argument is not provided, you will be prompted to enter information interactively",
        )

        self.parser.add_argument(
            "--template",
            type=str,
            default=None,
            help="Name of the example template config to use for build. You may use `llama stack build --list-templates` to check out the available templates",
        )

        self.parser.add_argument(
            "--list-templates",
            type=bool,
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Show the available templates for building a Llama Stack distribution",
        )

        self.parser.add_argument(
            "--name",
            type=str,
            help="Name of the Llama Stack build to override from template config. This name will be used as paths to store configuration files, build conda environments/docker images. If not specified, will use the name from the template config. ",
        )

        self.parser.add_argument(
            "--image-type",
            type=str,
            help="Image Type to use for the build. This can be either conda or docker. If not specified, will use the image type from the template config.",
            choices=["conda", "docker"],
        )

    def _get_build_config_from_name(self, args: argparse.Namespace) -> Optional[Path]:
        if os.getenv("CONDA_PREFIX", ""):
            conda_dir = (
                Path(os.getenv("CONDA_PREFIX")).parent / f"llamastack-{args.name}"
            )
        else:
            cprint(
                "Cannot find CONDA_PREFIX. Trying default conda path ~/.conda/envs...",
                color="green",
            )
            conda_dir = (
                Path(os.path.expanduser("~/.conda/envs")) / f"llamastack-{args.name}"
            )
        build_config_file = Path(conda_dir) / f"{args.name}-build.yaml"
        if build_config_file.exists():
            return build_config_file

        return None

    def _run_stack_build_command_from_build_config(
        self, build_config: BuildConfig
    ) -> None:
        import json
        import os

        import yaml

        from llama_stack.distribution.build import ApiInput, build_image, ImageType

        from llama_stack.distribution.utils.config_dirs import DISTRIBS_BASE_DIR
        from llama_stack.distribution.utils.serialize import EnumEncoder
        from termcolor import cprint

        # save build.yaml spec for building same distribution again
        if build_config.image_type == ImageType.docker.value:
            # docker needs build file to be in the llama-stack repo dir to be able to copy over to the image
            llama_stack_path = Path(
                os.path.abspath(__file__)
            ).parent.parent.parent.parent
            build_dir = llama_stack_path / "tmp/configs/"
        else:
            build_dir = DISTRIBS_BASE_DIR / f"llamastack-{build_config.name}"

        os.makedirs(build_dir, exist_ok=True)
        build_file_path = build_dir / f"{build_config.name}-build.yaml"

        with open(build_file_path, "w") as f:
            to_write = json.loads(json.dumps(build_config.dict(), cls=EnumEncoder))
            f.write(yaml.dump(to_write, sort_keys=False))

        return_code = build_image(build_config, build_file_path)
        if return_code != 0:
            return

        configure_name = (
            build_config.name
            if build_config.image_type == "conda"
            else (f"llamastack-{build_config.name}")
        )
        if build_config.image_type == "conda":
            cprint(
                f"You can now run `llama stack configure {configure_name}`",
                color="green",
            )
        else:
            cprint(
                f"You can now run `llama stack run {build_config.name}`",
                color="green",
            )

    def _run_template_list_cmd(self, args: argparse.Namespace) -> None:
        import json

        import yaml

        from llama_stack.cli.table import print_table

        # eventually, this should query a registry at llama.meta.com/llamastack/distributions
        headers = [
            "Template Name",
            "Providers",
            "Description",
        ]

        rows = []
        for spec in available_templates_specs():
            rows.append(
                [
                    spec.name,
                    json.dumps(spec.distribution_spec.providers, indent=2),
                    spec.distribution_spec.description,
                ]
            )
        print_table(
            rows,
            headers,
            separate_rows=True,
        )

    def _run_stack_build_command(self, args: argparse.Namespace) -> None:
        import yaml
        from llama_stack.distribution.distribution import get_provider_registry
        from prompt_toolkit import prompt
        from prompt_toolkit.validation import Validator
        from termcolor import cprint

        if args.list_templates:
            self._run_template_list_cmd(args)
            return

        if args.template:
            if not args.name:
                self.parser.error(
                    "You must specify a name for the build using --name when using a template"
                )
                return
            build_path = TEMPLATES_PATH / f"{args.template}-build.yaml"
            if not build_path.exists():
                self.parser.error(
                    f"Could not find template {args.template}. Please run `llama stack build --list-templates` to check out the available templates"
                )
                return
            with open(build_path, "r") as f:
                build_config = BuildConfig(**yaml.safe_load(f))
                build_config.name = args.name
                if args.image_type:
                    build_config.image_type = args.image_type
                self._run_stack_build_command_from_build_config(build_config)

            return

        # try to see if we can find a pre-existing build config file through name
        if args.name:
            maybe_build_config = self._get_build_config_from_name(args)
            if maybe_build_config:
                cprint(
                    f"Building from existing build config for {args.name} in {str(maybe_build_config)}...",
                    "green",
                )
                with open(maybe_build_config, "r") as f:
                    build_config = BuildConfig(**yaml.safe_load(f))
                    self._run_stack_build_command_from_build_config(build_config)
                    return

        if not args.config and not args.template:
            if not args.name:
                name = prompt(
                    "> Enter a name for your Llama Stack (e.g. my-local-stack): ",
                    validator=Validator.from_callable(
                        lambda x: len(x) > 0,
                        error_message="Name cannot be empty, please enter a name",
                    ),
                )
            else:
                name = args.name

            image_type = prompt(
                "> Enter the image type you want your Llama Stack to be built as (docker or conda): ",
                validator=Validator.from_callable(
                    lambda x: x in ["docker", "conda"],
                    error_message="Invalid image type, please enter conda or docker",
                ),
                default="conda",
            )

            cprint(
                "\n Llama Stack is composed of several APIs working together. Let's configure the providers (implementations) you want to use for these APIs.",
                color="green",
            )

            providers = dict()
            for api, providers_for_api in get_provider_registry().items():
                api_provider = prompt(
                    "> Enter provider for the {} API: (default=meta-reference): ".format(
                        api.value
                    ),
                    validator=Validator.from_callable(
                        lambda x: x in providers_for_api,
                        error_message="Invalid provider, please enter one of the following: {}".format(
                            list(providers_for_api.keys())
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
                "\n > (Optional) Enter a short description for your Llama Stack: ",
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
            self._run_stack_build_command_from_build_config(build_config)
