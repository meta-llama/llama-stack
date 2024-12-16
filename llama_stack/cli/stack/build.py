# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_stack.cli.subcommand import Subcommand
from llama_stack.distribution.datatypes import *  # noqa: F403
import os
import shutil
from functools import lru_cache
from pathlib import Path

import pkg_resources

from llama_stack.distribution.distribution import get_provider_registry
from llama_stack.distribution.resolver import InvalidProviderError
from llama_stack.distribution.utils.dynamic import instantiate_class_type

TEMPLATES_PATH = Path(__file__).parent.parent.parent / "templates"


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
            help="Path to a config file to use for the build. You can find example configs in llama_stack/distribution/**/build.yaml. If this argument is not provided, you will be prompted to enter information interactively",
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
            choices=["conda", "docker", "venv"],
            default="conda",
        )

        self.parser.add_argument(
            "--platform",
            type=str,
            default=None,
            help="Platform to use for the build. Required when using docker as image type, defaults to host if no platform is specified",
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
                    if args.platform:
                        build_config.platform = args.platform
                    if args.image_type:
                        build_config.image_type = args.image_type
                    else:
                        self.parser.error(
                            f"Please specify a image-type (docker | conda) for {args.template}"
                        )
                    self._run_stack_build_command_from_build_config(
                        build_config, template_name=args.template
                    )
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
                    lambda x: x in ["docker", "conda", "venv"],
                    error_message="Invalid image type, please enter conda or docker or venv",
                ),
                default="conda",
            )

            platform = prompt(
                "> Enter the target platform you want your Llama Stack to be built for: "
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
            if platform.strip():
                build_config.platform = platform
            self._run_stack_build_command_from_build_config(build_config)
            return

        with open(args.config, "r") as f:
            try:
                build_config = BuildConfig(**yaml.safe_load(f))
            except Exception as e:
                self.parser.error(f"Could not parse config file {args.config}: {e}")
                return
            self._run_stack_build_command_from_build_config(build_config)

    def _generate_run_config(self, build_config: BuildConfig, build_dir: Path) -> None:
        """
        Generate a run.yaml template file for user to edit from a build.yaml file
        """
        import json

        import yaml
        from termcolor import cprint

        from llama_stack.distribution.build import ImageType

        apis = list(build_config.distribution_spec.providers.keys())
        run_config = StackRunConfig(
            docker_image=(
                build_config.name
                if build_config.image_type == ImageType.docker.value
                else None
            ),
            image_name=build_config.name,
            conda_env=(
                build_config.name
                if build_config.image_type == ImageType.conda.value
                else None
            ),
            apis=apis,
            providers={},
        )
        # build providers dict
        provider_registry = get_provider_registry()
        for api in apis:
            run_config.providers[api] = []
            provider_types = build_config.distribution_spec.providers[api]
            if isinstance(provider_types, str):
                provider_types = [provider_types]

            for i, provider_type in enumerate(provider_types):
                pid = provider_type.split("::")[-1]

                p = provider_registry[Api(api)][provider_type]
                if p.deprecation_error:
                    raise InvalidProviderError(p.deprecation_error)

                config_type = instantiate_class_type(
                    provider_registry[Api(api)][provider_type].config_class
                )
                if hasattr(config_type, "sample_run_config"):
                    config = config_type.sample_run_config(
                        __distro_dir__=f"distributions/{build_config.name}"
                    )
                else:
                    config = {}

                p_spec = Provider(
                    provider_id=f"{pid}-{i}" if len(provider_types) > 1 else pid,
                    provider_type=provider_type,
                    config=config,
                )
                run_config.providers[api].append(p_spec)

        os.makedirs(build_dir, exist_ok=True)
        run_config_file = build_dir / f"{build_config.name}-run.yaml"

        with open(run_config_file, "w") as f:
            to_write = json.loads(run_config.model_dump_json())
            f.write(yaml.dump(to_write, sort_keys=False))

        cprint(
            f"You can now edit {run_config_file} and run `llama stack run {run_config_file}`",
            color="green",
        )

    def _run_stack_build_command_from_build_config(
        self, build_config: BuildConfig, template_name: Optional[str] = None
    ) -> None:
        import json
        import os

        import yaml
        from termcolor import cprint

        from llama_stack.distribution.build import build_image
        from llama_stack.distribution.utils.config_dirs import DISTRIBS_BASE_DIR

        # save build.yaml spec for building same distribution again
        build_dir = DISTRIBS_BASE_DIR / f"llamastack-{build_config.name}"
        os.makedirs(build_dir, exist_ok=True)
        build_file_path = build_dir / f"{build_config.name}-build.yaml"

        with open(build_file_path, "w") as f:
            to_write = json.loads(build_config.model_dump_json())
            f.write(yaml.dump(to_write, sort_keys=False))

        return_code = build_image(build_config, build_file_path)
        if return_code != 0:
            return

        if template_name:
            # copy run.yaml from template to build_dir instead of generating it again
            template_path = pkg_resources.resource_filename(
                "llama_stack", f"templates/{template_name}/run.yaml"
            )
            os.makedirs(build_dir, exist_ok=True)
            run_config_file = build_dir / f"{build_config.name}-run.yaml"
            shutil.copy(template_path, run_config_file)

            # Find all ${env.VARIABLE} patterns
            cprint("Build Successful!", color="green")
        else:
            self._generate_run_config(build_config, build_dir)

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
