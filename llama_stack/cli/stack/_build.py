# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import importlib.resources
import json
import os
import shutil
import sys
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import yaml
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.validation import Validator
from termcolor import cprint

from llama_stack.cli.table import print_table
from llama_stack.distribution.build import (
    SERVER_DEPENDENCIES,
    build_image,
    get_provider_dependencies,
)
from llama_stack.distribution.configure import parse_and_maybe_upgrade_config
from llama_stack.distribution.datatypes import (
    BuildConfig,
    DistributionSpec,
    Provider,
    StackRunConfig,
)
from llama_stack.distribution.distribution import get_provider_registry
from llama_stack.distribution.resolver import InvalidProviderError
from llama_stack.distribution.utils.config_dirs import DISTRIBS_BASE_DIR
from llama_stack.distribution.utils.dynamic import instantiate_class_type
from llama_stack.distribution.utils.exec import formulate_run_args, run_command
from llama_stack.distribution.utils.image_types import LlamaStackImageType
from llama_stack.providers.datatypes import Api

TEMPLATES_PATH = Path(__file__).parent.parent.parent / "templates"


@lru_cache()
def available_templates_specs() -> Dict[str, BuildConfig]:
    import yaml

    template_specs = {}
    for p in TEMPLATES_PATH.rglob("*build.yaml"):
        template_name = p.parent.name
        with open(p, "r") as f:
            build_config = BuildConfig(**yaml.safe_load(f))
            template_specs[template_name] = build_config
    return template_specs


def run_stack_build_command(args: argparse.Namespace) -> None:
    if args.list_templates:
        return _run_template_list_cmd()

    if args.image_type == "venv":
        current_venv = os.environ.get("VIRTUAL_ENV")
        image_name = args.image_name or current_venv
    elif args.image_type == "conda":
        current_conda_env = os.environ.get("CONDA_DEFAULT_ENV")
        image_name = args.image_name or current_conda_env
    else:
        image_name = args.image_name

    if args.template:
        available_templates = available_templates_specs()
        if args.template not in available_templates:
            cprint(
                f"Could not find template {args.template}. Please run `llama stack build --list-templates` to check out the available templates",
                color="red",
            )
            sys.exit(1)
        build_config = available_templates[args.template]
        if args.image_type:
            build_config.image_type = args.image_type
        else:
            cprint(
                f"Please specify a image-type (container | conda | venv) for {args.template}",
                color="red",
            )
            sys.exit(1)
    elif not args.config and not args.template:
        name = prompt(
            "> Enter a name for your Llama Stack (e.g. my-local-stack): ",
            validator=Validator.from_callable(
                lambda x: len(x) > 0,
                error_message="Name cannot be empty, please enter a name",
            ),
        )

        image_type = prompt(
            "> Enter the image type you want your Llama Stack to be built as (container or conda or venv): ",
            validator=Validator.from_callable(
                lambda x: x in ["container", "conda", "venv"],
                error_message="Invalid image type, please enter conda or container or venv",
            ),
            default="conda",
        )

        if image_type == "conda":
            if not image_name:
                cprint(
                    f"No current conda environment detected or specified, will create a new conda environment with the name `llamastack-{name}`",
                    color="yellow",
                )
                image_name = f"llamastack-{name}"
            else:
                cprint(
                    f"Using conda environment {image_name}",
                    color="green",
                )
        else:
            image_name = f"llamastack-{name}"

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
            available_providers = [x for x in providers_for_api.keys() if x not in ("remote", "remote::sample")]
            api_provider = prompt(
                "> Enter provider for API {}: ".format(api.value),
                completer=WordCompleter(available_providers),
                complete_while_typing=True,
                validator=Validator.from_callable(
                    lambda x: x in available_providers,  # noqa: B023 - see https://github.com/astral-sh/ruff/issues/7847
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

        build_config = BuildConfig(image_type=image_type, distribution_spec=distribution_spec)
    else:
        with open(args.config, "r") as f:
            try:
                build_config = BuildConfig(**yaml.safe_load(f))
            except Exception as e:
                cprint(
                    f"Could not parse config file {args.config}: {e}",
                    color="red",
                )
                sys.exit(1)

        if build_config.image_type == LlamaStackImageType.CONTAINER.value and not args.image_name:
            cprint(
                "Please specify --image-name when building a container from a config file",
                color="red",
            )
            sys.exit(1)

    if args.print_deps_only:
        print(f"# Dependencies for {args.template or args.config or image_name}")
        normal_deps, special_deps = get_provider_dependencies(build_config.distribution_spec.providers)
        normal_deps += SERVER_DEPENDENCIES
        print(f"uv pip install {' '.join(normal_deps)}")
        for special_dep in special_deps:
            print(f"uv pip install {special_dep}")
        return

    try:
        run_config = _run_stack_build_command_from_build_config(
            build_config,
            image_name=image_name,
            config_path=args.config,
            template_name=args.template,
        )

    except (Exception, RuntimeError) as exc:
        cprint(
            f"Error building stack: {exc}",
            color="red",
        )
        sys.exit(1)
    if run_config is None:
        cprint(
            "Run config path is empty",
            color="red",
        )
        sys.exit(1)

    if args.run:
        run_config = Path(run_config)
        config_dict = yaml.safe_load(run_config.read_text())
        config = parse_and_maybe_upgrade_config(config_dict)
        run_args = formulate_run_args(args.image_type, args.image_name, config, args.template)
        run_args.extend([run_config, str(os.getenv("LLAMA_STACK_PORT", 8321))])
        run_command(run_args)


def _generate_run_config(
    build_config: BuildConfig,
    build_dir: Path,
    image_name: str,
) -> str:
    """
    Generate a run.yaml template file for user to edit from a build.yaml file
    """
    apis = list(build_config.distribution_spec.providers.keys())
    run_config = StackRunConfig(
        container_image=(image_name if build_config.image_type == LlamaStackImageType.CONTAINER.value else None),
        image_name=image_name,
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

            config_type = instantiate_class_type(provider_registry[Api(api)][provider_type].config_class)
            if hasattr(config_type, "sample_run_config"):
                config = config_type.sample_run_config(__distro_dir__=f"~/.llama/distributions/{image_name}")
            else:
                config = {}

            p_spec = Provider(
                provider_id=f"{pid}-{i}" if len(provider_types) > 1 else pid,
                provider_type=provider_type,
                config=config,
            )
            run_config.providers[api].append(p_spec)

    run_config_file = build_dir / f"{image_name}-run.yaml"

    with open(run_config_file, "w") as f:
        to_write = json.loads(run_config.model_dump_json())
        f.write(yaml.dump(to_write, sort_keys=False))

    # this path is only invoked when no template is provided
    cprint(
        f"You can now run your stack with `llama stack run {run_config_file}`",
        color="green",
    )
    return run_config_file


def _run_stack_build_command_from_build_config(
    build_config: BuildConfig,
    image_name: Optional[str] = None,
    template_name: Optional[str] = None,
    config_path: Optional[str] = None,
) -> str:
    if build_config.image_type == LlamaStackImageType.CONTAINER.value:
        if template_name:
            image_name = f"distribution-{template_name}"
        else:
            if not image_name:
                raise ValueError("Please specify an image name when building a container image without a template")
    elif build_config.image_type == LlamaStackImageType.CONDA.value:
        if not image_name:
            raise ValueError("Please specify an image name when building a conda image")
    elif build_config.image_type == LlamaStackImageType.VENV.value:
        if not image_name and os.environ.get("UV_SYSTEM_PYTHON"):
            image_name = "__system__"
        if not image_name:
            raise ValueError("Please specify an image name when building a venv image")

    if template_name:
        build_dir = DISTRIBS_BASE_DIR / template_name
        build_file_path = build_dir / f"{template_name}-build.yaml"
    else:
        build_dir = DISTRIBS_BASE_DIR / image_name
        build_file_path = build_dir / f"{image_name}-build.yaml"

    os.makedirs(build_dir, exist_ok=True)
    with open(build_file_path, "w") as f:
        to_write = json.loads(build_config.model_dump_json())
        f.write(yaml.dump(to_write, sort_keys=False))

    return_code = build_image(
        build_config,
        build_file_path,
        image_name,
        template_or_config=template_name or config_path,
    )
    if return_code != 0:
        raise RuntimeError(f"Failed to build image {image_name}")

    if template_name:
        # copy run.yaml from template to build_dir instead of generating it again
        template_path = importlib.resources.files("llama_stack") / f"templates/{template_name}/run.yaml"
        with importlib.resources.as_file(template_path) as path:
            run_config_file = build_dir / f"{template_name}-run.yaml"
            shutil.copy(path, run_config_file)

        cprint("Build Successful!", color="green")
        return template_path
    else:
        return _generate_run_config(build_config, build_dir, image_name)


def _run_template_list_cmd() -> None:
    # eventually, this should query a registry at llama.meta.com/llamastack/distributions
    headers = [
        "Template Name",
        # "Providers",
        "Description",
    ]

    rows = []
    for template_name, spec in available_templates_specs().items():
        rows.append(
            [
                template_name,
                # json.dumps(spec.distribution_spec.providers, indent=2),
                spec.distribution_spec.description,
            ]
        )
    print_table(
        rows,
        headers,
        separate_rows=True,
    )
