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

TEMPLATES_PATH = Path(os.path.relpath(__file__)).parent.parent.parent / "templates"


@lru_cache()
def available_templates_specs() -> List[BuildConfig]:
    import yaml

    template_specs = []
    for p in TEMPLATES_PATH.rglob("*build.yaml"):
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
            "--image-type",
            type=str,
            help="Image Type to use for the build. This can be either conda or docker. If not specified, will use the image type from the template config.",
            choices=["conda", "docker"],
            default="conda",
        )

    def _run_stack_build_command(self, args: argparse.Namespace) -> None:
        import textwrap

        import yaml
        from prompt_toolkit import prompt
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.validation import Validator
        from termcolor import cprint

        from llama_stack.distribution.distribution import get_provider_registry

        if args.list_templates:
            self._run_template_list_cmd(args)
            return

        if args.template:
            available_templates = available_templates_specs()
            for build_config in available_templates:
                if build_config.name == args.template:
                    if args.image_type:
                        build_config.image_type = args.image_type
                    else:
                        self.parser.error(
                            f"Please specify a image-type (docker | conda) for {args.template}"
                        )
                    self._run_stack_build_command_from_build_config(build_config)
                    return

            self.parser.error(
                f"Could not find template {args.template}. Please run `llama stack build --list-templates` to check out the available templates"
            )
            return

        if not args.config and not args.template:
            name = prompt(
                "> Enter a name for your Llama Stack (e.g. my-local-stack): ",
                validator=Validator.from_callable(
                    lambda x: len(x) > 0,
                    error_message="Name cannot be empty, please enter a name",
                ),
            )

            image_type = prompt(
                "> Enter the image type you want your Llama Stack to be built as (docker or conda): ",
                validator=Validator.from_callable(
                    lambda x: x in ["docker", "conda"],
                    error_message="Invalid image type, please enter conda or docker",
                ),
                default="conda",
            )

            cprint(
                textwrap.dedent(
                    """
                Llama Stack is composed of several APIs working together. Let's select
                the provider types (implementations) you want to use for these APIs.
                """,
                ),
                color="green",
            )

            print("Tip: use <TAB> to see options for the providers.\n")

            providers = dict()
            for api, providers_for_api in get_provider_registry().items():
                available_providers = [
                    x
                    for x in providers_for_api.keys()
                    if x not in ("remote", "remote::sample")
                ]
                api_provider = prompt(
                    "> Enter provider for API {}: ".format(api.value),
                    completer=WordCompleter(available_providers),
                    complete_while_typing=True,
                    validator=Validator.from_callable(
                        lambda x: x in available_providers,
                        error_message="Invalid provider, use <TAB> to see options",
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

    def _run_stack_build_command_from_build_config(
        self, build_config: BuildConfig
    ) -> None:
        import json
        import os

        import yaml
        from termcolor import cprint

        from llama_stack.distribution.build import build_image, ImageType
        from llama_stack.distribution.utils.config_dirs import DISTRIBS_BASE_DIR
        from llama_stack.distribution.utils.serialize import EnumEncoder

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
                f"You can now edit your run.yaml file and run `docker run -it -p 5000:5000 {build_config.name}`. See full command in llama-stack/distributions/",
                color="green",
            )

    def _run_template_list_cmd(self, args: argparse.Namespace) -> None:
        import json

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
