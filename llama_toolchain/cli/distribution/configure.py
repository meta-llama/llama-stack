# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import importlib
import inspect
import shlex

from pathlib import Path
from typing import Annotated, get_args, get_origin, Literal, Union

import yaml
from pydantic import BaseModel
from termcolor import cprint

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.distribution.datatypes import Distribution, PassthroughApiAdapter
from llama_toolchain.distribution.registry import available_distributions
from llama_toolchain.utils import DISTRIBS_BASE_DIR
from .utils import run_command

DISTRIBS = available_distributions()


class DistributionConfigure(Subcommand):
    """Llama cli for configuring llama toolchain configs"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "configure",
            prog="llama distribution configure",
            description="configure a llama stack distribution",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_distribution_configure_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "--name",
            type=str,
            help="Mame of the distribution to configure",
            default="local-source",
            choices=[d.name for d in available_distributions()],
        )

    def _run_distribution_configure_cmd(self, args: argparse.Namespace) -> None:
        dist = None
        for d in DISTRIBS:
            if d.name == args.name:
                dist = d
                break

        if dist is None:
            self.parser.error(f"Could not find distribution {args.name}")
            return

        env_file = DISTRIBS_BASE_DIR / dist.name / "conda.env"
        # read this file to get the conda env name
        assert env_file.exists(), f"Could not find conda env file {env_file}"
        with open(env_file, "r") as f:
            conda_env = f.read().strip()

        configure_llama_distribution(dist, conda_env)


def configure_llama_distribution(dist: Distribution, conda_env: str):
    python_exe = run_command(shlex.split("which python"))
    # simple check
    if conda_env not in python_exe:
        raise ValueError(
            f"Please re-run configure by activating the `{conda_env}` conda environment"
        )

    adapter_configs = {}
    for api_surface, adapter in dist.adapters.items():
        if isinstance(adapter, PassthroughApiAdapter):
            adapter_configs[api_surface.value] = adapter.dict()
        else:
            cprint(
                f"Configuring API surface: {api_surface.value}", "white", attrs=["bold"]
            )
            config_type = instantiate_class_type(adapter.config_class)
            # TODO: when we are re-configuring, we should read existing values
            config = prompt_for_config(config_type)
            adapter_configs[api_surface.value] = config.dict()

    dist_config = {
        "adapters": adapter_configs,
        "conda_env": conda_env,
    }

    yaml_output_path = Path(DISTRIBS_BASE_DIR) / dist.name / "config.yaml"
    with open(yaml_output_path, "w") as fp:
        fp.write(yaml.dump(dist_config, sort_keys=False))

    print(f"YAML configuration has been written to {yaml_output_path}")


def instantiate_class_type(fully_qualified_name):
    module_name, class_name = fully_qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_literal_values(field):
    """Extract literal values from a field if it's a Literal type."""
    if get_origin(field.annotation) is Literal:
        return get_args(field.annotation)
    return None


def is_optional(field_type):
    """Check if a field type is Optional."""
    return get_origin(field_type) is Union and type(None) in get_args(field_type)


def get_non_none_type(field_type):
    """Get the non-None type from an Optional type."""
    return next(arg for arg in get_args(field_type) if arg is not type(None))


def prompt_for_config(config_type: type[BaseModel]) -> BaseModel:
    """
    Recursively prompt the user for configuration values based on a Pydantic BaseModel.

    Args:
        config_type: A Pydantic BaseModel class representing the configuration structure.

    Returns:
        An instance of the config_type with user-provided values.
    """
    config_data = {}

    for field_name, field in config_type.__fields__.items():
        field_type = field.annotation
        default_value = (
            field.default if not isinstance(field.default, type(Ellipsis)) else None
        )
        is_required = field.required

        # Skip fields with Literal type
        if get_origin(field_type) is Literal:
            continue

        # Check if the field is a discriminated union
        if get_origin(field_type) is Annotated:
            inner_type = get_args(field_type)[0]
            if get_origin(inner_type) is Union:
                discriminator = field.field_info.discriminator
                if discriminator:
                    union_types = get_args(inner_type)
                    # Find the discriminator field in each union type
                    type_map = {}
                    for t in union_types:
                        disc_field = t.__fields__[discriminator]
                        literal_values = get_literal_values(disc_field)
                        if literal_values:
                            for value in literal_values:
                                type_map[value] = t

                    while True:
                        discriminator_value = input(
                            f"Enter the {discriminator} (options: {', '.join(type_map.keys())}): "
                        )
                        if discriminator_value in type_map:
                            chosen_type = type_map[discriminator_value]
                            print(f"\nConfiguring {chosen_type.__name__}:")
                            sub_config = prompt_for_config(chosen_type)
                            config_data[field_name] = sub_config
                            # Set the discriminator field in the sub-config
                            setattr(sub_config, discriminator, discriminator_value)
                            break
                        else:
                            print(f"Invalid {discriminator}. Please try again.")
                    continue

        if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
            print(f"\nEntering sub-configuration for {field_name}:")
            config_data[field_name] = prompt_for_config(field_type)
        else:
            prompt = f"Enter value for {field_name}"
            if default_value is not None:
                prompt += f" (default: {default_value})"
            if is_optional(field_type):
                prompt += " (optional)"
            elif is_required:
                prompt += " (required)"
            prompt += ": "

            while True:
                user_input = input(prompt)
                if user_input == "":
                    if default_value is not None:
                        config_data[field_name] = default_value
                        break
                    elif is_optional(field_type):
                        config_data[field_name] = None
                        break
                    elif not is_required:
                        config_data[field_name] = None
                        break
                    else:
                        print("This field is required. Please provide a value.")
                        continue

                try:
                    # Handle Optional types
                    if is_optional(field_type):
                        if user_input.lower() == "none":
                            config_data[field_name] = None
                            break
                        field_type = get_non_none_type(field_type)

                    # Convert the input to the correct type
                    if inspect.isclass(field_type) and issubclass(
                        field_type, BaseModel
                    ):
                        # For nested BaseModels, we assume a dictionary-like string input
                        import ast

                        config_data[field_name] = field_type(
                            **ast.literal_eval(user_input)
                        )
                    else:
                        config_data[field_name] = field_type(user_input)
                    break
                except ValueError:
                    print(
                        f"Invalid input. Expected type: {getattr(field_type, '__name__', str(field_type))}"
                    )

    return config_type(**config_data)
