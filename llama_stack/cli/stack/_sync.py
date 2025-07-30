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
from pathlib import Path

import yaml
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.validation import Validator
from termcolor import colored, cprint

from llama_stack.cli.stack.utils import ImageType, available_templates_specs, generate_run_config
from llama_stack.core.build import get_provider_dependencies
from llama_stack.core.datatypes import (
    BuildConfig,
    BuildProvider,
    DistributionSpec,
)
from llama_stack.core.distribution import get_provider_registry
from llama_stack.core.external import load_external_apis
from llama_stack.core.stack import replace_env_vars
from llama_stack.core.utils.config_dirs import DISTRIBS_BASE_DIR
from llama_stack.core.utils.exec import run_command
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import Api

TEMPLATES_PATH = Path(__file__).parent.parent.parent / "templates"

logger = get_logger(name=__name__, category="cli")


# These are the dependencies needed by the distribution server.
# `llama-stack` is automatically installed by the installation script.
SERVER_DEPENDENCIES = [
    "aiosqlite",
    "fastapi",
    "fire",
    "httpx",
    "uvicorn",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp-proto-http",
]


def run_stack_sync_command(args: argparse.Namespace) -> None:
    current_venv = os.environ.get("VIRTUAL_ENV")
    env_name = args.env_name or current_venv

    if args.template:
        available_templates = available_templates_specs()
        if args.template not in available_templates:
            cprint(
                f"Could not find template {args.template}. Please run `llama stack sync --list-templates` to check out the available templates",
                color="red",
                file=sys.stderr,
            )
            sys.exit(1)
        build_config = available_templates[args.template]
        # always venv, conda is gone and container is separate.
        build_config.image_type = ImageType.VENV.value
    elif args.providers:
        provider_list: dict[str, list[BuildProvider]] = dict()
        for api_provider in args.providers.split(","):
            if "=" not in api_provider:
                cprint(
                    "Could not parse `--providers`. Please ensure the list is in the format api1=provider1,api2=provider2",
                    color="red",
                    file=sys.stderr,
                )
                sys.exit(1)
            api, provider_type = api_provider.split("=")
            providers_for_api = get_provider_registry().get(Api(api), None)
            if providers_for_api is None:
                cprint(
                    f"{api} is not a valid API.",
                    color="red",
                    file=sys.stderr,
                )
                sys.exit(1)
            if provider_type in providers_for_api:
                provider = BuildProvider(
                    provider_type=provider_type,
                    module=None,
                )
                provider_list.setdefault(api, []).append(provider)
            else:
                cprint(
                    f"{provider} is not a valid provider for the {api} API.",
                    color="red",
                    file=sys.stderr,
                )
                sys.exit(1)
        distribution_spec = DistributionSpec(
            providers=provider_list,
            description=",".join(args.providers),
        )
        build_config = BuildConfig(image_type=ImageType.VENV.value, distribution_spec=distribution_spec)
    elif not args.config and not args.template:
        name = prompt(
            "> Enter a name for your Llama Stack (e.g. my-local-stack): ",
            validator=Validator.from_callable(
                lambda x: len(x) > 0,
                error_message="Name cannot be empty, please enter a name",
            ),
        )

        image_type = prompt(
            "> Enter the image type you want your Llama Stack to be built as (use <TAB> to see options): ",
            completer=WordCompleter([e.value for e in ImageType]),
            complete_while_typing=True,
            validator=Validator.from_callable(
                lambda x: x in [e.value for e in ImageType],
                error_message="Invalid image type. Use <TAB> to see options",
            ),
        )

        env_name = f"llamastack-{name}"

        cprint(
            textwrap.dedent(
                """
            Llama Stack is composed of several APIs working together. Let's select
            the provider types (implementations) you want to use for these APIs.
            """,
            ),
            color="green",
            file=sys.stderr,
        )

        cprint("Tip: use <TAB> to see options for the providers.\n", color="green", file=sys.stderr)

        providers: dict[str, list[BuildProvider]] = dict()
        for api, providers_for_api in get_provider_registry().items():
            available_providers = [x for x in providers_for_api.keys() if x not in ("remote", "remote::sample")]
            if not available_providers:
                continue
            api_provider = prompt(
                f"> Enter provider for API {api.value}: ",
                completer=WordCompleter(available_providers),
                complete_while_typing=True,
                validator=Validator.from_callable(
                    lambda x: x in available_providers,  # noqa: B023 - see https://github.com/astral-sh/ruff/issues/7847
                    error_message="Invalid provider, use <TAB> to see options",
                ),
            )

            string_providers = api_provider.split(" ")

            for provider in string_providers:
                providers.setdefault(api.value, []).append(BuildProvider(provider_type=provider))

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
        with open(args.config) as f:
            try:
                contents = yaml.safe_load(f)
                contents = replace_env_vars(contents)
                build_config = BuildConfig(**contents)
                build_config.image_type = "venv"
            except Exception as e:
                cprint(
                    f"Could not parse config file {args.config}: {e}",
                    color="red",
                    file=sys.stderr,
                )
                sys.exit(1)

    if args.print_deps_only:
        print(f"# Dependencies for {args.template or args.config or env_name}")
        normal_deps, special_deps, external_provider_dependencies = get_provider_dependencies(build_config)
        normal_deps += SERVER_DEPENDENCIES
        print(f"uv pip install {' '.join(normal_deps)}")
        for special_dep in special_deps:
            print(f"uv pip install {special_dep}")
        for external_dep in external_provider_dependencies:
            print(f"uv pip install {external_dep}")
        return

    try:
        # just uv pip install
        run_config = _run_stack_sync_command_from_build_config(
            build_config,
            env_name=env_name,
            config_path=args.config,
            template_name=args.template,
        )

    except (Exception, RuntimeError) as exc:
        import traceback

        cprint(
            f"Error syncing stack: {exc}",
            color="red",
            file=sys.stderr,
        )
        cprint("Stack trace:", color="red", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    if run_config is None:
        cprint(
            "Run config path is empty",
            color="red",
            file=sys.stderr,
        )
        sys.exit(1)


def _run_stack_sync_command_from_build_config(
    build_config: BuildConfig,
    env_name: str | None = None,
    template_name: str | None = None,
    config_path: str | None = None,
):
    if not env_name and os.environ.get("UV_SYSTEM_PYTHON"):
        env_name = "__system__"
    else:
        env_name = env_name or build_config.image_name
        if os.environ.get("VIRTUAL_ENV") == env_name:
            logger.info("using current venv as its the same as --env-name")
        else:
            if env_name is None:
                raise ValueError("env_name cannot be None")
            return_code = run_command(["uv", "venv", env_name])
            run_command(["source", f"{env_name}/bin/activate"])

    if template_name:
        build_dir = DISTRIBS_BASE_DIR / template_name
        build_file_path = build_dir / f"{template_name}-build.yaml"
    else:
        if env_name is None:
            raise ValueError("env_name cannot be None")
        build_dir = DISTRIBS_BASE_DIR / env_name
        build_file_path = build_dir / f"{env_name}-build.yaml"

    os.makedirs(build_dir, exist_ok=True)

    with open(build_file_path, "w") as f:
        to_write = json.loads(build_config.model_dump_json(exclude_none=True))
        f.write(yaml.dump(to_write, sort_keys=False))

    # We first install the external APIs so that the build process can use them and discover the
    # providers dependencies
    if build_config.external_apis_dir:
        cprint("Installing external APIs", color="yellow", file=sys.stderr)
        external_apis = load_external_apis(build_config)
        if external_apis:
            # install the external APIs
            packages = []
            for _, api_spec in external_apis.items():
                if api_spec.pip_packages:
                    packages.extend(api_spec.pip_packages)
                    cprint(
                        f"Installing {api_spec.name} with pip packages {api_spec.pip_packages}",
                        color="yellow",
                        file=sys.stderr,
                    )
            return_code = run_command(["uv", "pip", "install", "--no-cache-dir", *packages])
            if return_code != 0:
                packages_str = ", ".join(packages)
                raise RuntimeError(
                    f"Failed to install external APIs packages: {packages_str} (return code: {return_code})"
                )

    registry = get_provider_registry(build_config)
    for api, providers in build_config.distribution_spec.providers.items():
        providers_for_api = registry[Api(api)]
        for provider in providers:
            if provider.provider_type not in providers_for_api:
                raise ValueError(f"Provider `{provider}` is not available for API `{api}`")
            # install the external provider module if available
            if hasattr(provider, "module") and provider.module is not None:
                return_code = run_command(["uv", "pip", "install", "--no-cache-dir", *provider.module])
                if return_code != 0:
                    packages_str = ", ".join(packages)
                    raise RuntimeError(
                        f"Failed to install external APIs packages: {packages_str} (return code: {return_code})"
                    )
            mod = mod = providers_for_api[provider.provider_type].module
            if mod is None:
                raise ValueError("provider.module cannot be None")
            mod = mod.replace(".", "/")
            return_code = run_command(["uv", "pip", "install", "--no-cache-dir", mod])
            if return_code != 0:
                packages_str = ", ".join(packages)
                raise RuntimeError(f"Failed to install API packages: {packages_str} (return code: {return_code})")
            return_code = run_command(["uv", "pip", "install", "--no-cache-dir", *SERVER_DEPENDENCIES])
            if return_code != 0:
                packages_str = ", ".join(packages)
                raise RuntimeError(f"Failed to install API packages: {packages_str} (return code: {return_code})")

    if template_name:
        # copy run.yaml from template to build_dir instead of generating it again
        template_path = importlib.resources.files("llama_stack") / f"templates/{template_name}/run.yaml"
        run_config_file = build_dir / f"{template_name}-run.yaml"

        with importlib.resources.as_file(template_path) as path:
            shutil.copy(path, run_config_file)

        cprint("Sync Successful!", color="green", file=sys.stderr)
        cprint(
            f"You can find a run config here: {run_config_file} to pass into `llama stack run`",
            color="blue",
            file=sys.stderr,
        )
        cprint(
            "You can run the new Llama Stack distro via: "
            + colored(f"llama stack run {run_config_file} --image-type {build_config.image_type}", "blue"),
            color="green",
            file=sys.stderr,
        )
        return template_path
    else:
        if env_name is None:
            raise ValueError("env_name cannot be None")
        return generate_run_config(build_config, build_dir, env_name)
