# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import inspect
import json
from enum import Enum

from typing import Any, get_args, get_origin, List, Literal, Optional, Type, Union

from pydantic import BaseModel
from pydantic_core import PydanticUndefinedType

from typing_extensions import Annotated


def is_list_of_primitives(field_type):
    """Check if a field type is a List of primitive types."""
    origin = get_origin(field_type)
    if origin is List or origin is list:
        args = get_args(field_type)
        if len(args) == 1 and args[0] in (int, float, str, bool):
            return True
    return False


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


def manually_validate_field(model: Type[BaseModel], field_name: str, value: Any):
    validators = model.__pydantic_decorators__.field_validators
    for _name, validator in validators.items():
        if field_name in validator.info.fields:
            validator.func(value)

    return value


# This is somewhat elaborate, but does not purport to be comprehensive in any way.
# We should add handling for the most common cases to tide us over.
#
# doesn't support List[nested_class] yet or Dicts of any kind. needs a bunch of
# unit tests for coverage.
def prompt_for_config(
    config_type: type[BaseModel], existing_config: Optional[BaseModel] = None
) -> BaseModel:
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

        existing_value = (
            getattr(existing_config, field_name) if existing_config else None
        )
        if existing_value:
            default_value = existing_value
        else:
            default_value = (
                field.default
                if not isinstance(field.default, PydanticUndefinedType)
                else None
            )
        is_required = field.is_required

        # Skip fields with Literal type
        if get_origin(field_type) is Literal:
            continue

        if inspect.isclass(field_type) and issubclass(field_type, Enum):
            prompt = f"Choose {field_name} (options: {', '.join(e.name for e in field_type)}):"
            while True:
                # this branch does not handle existing and default values yet
                user_input = input(prompt + " ")
                try:
                    value = field_type[user_input]
                    validated_value = manually_validate_field(config_type, field, value)
                    config_data[field_name] = validated_value
                    break
                except KeyError:
                    print(
                        f"Invalid choice. Please choose from: {', '.join(e.name for e in field_type)}"
                    )
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

                            if existing_value and (
                                getattr(existing_value, discriminator)
                                != discriminator_value
                            ):
                                existing_value = None

                            sub_config = prompt_for_config(chosen_type, existing_value)
                            config_data[field_name] = sub_config
                            # Set the discriminator field in the sub-config
                            setattr(sub_config, discriminator, discriminator_value)
                            break
                        else:
                            print(f"Invalid {discriminator}. Please try again.")
                    continue

        if (
            is_optional(field_type)
            and inspect.isclass(get_non_none_type(field_type))
            and issubclass(get_non_none_type(field_type), BaseModel)
        ):
            prompt = f"Do you want to configure {field_name}? (y/n): "
            if input(prompt).lower() == "n":
                config_data[field_name] = None
                continue
            nested_type = get_non_none_type(field_type)
            print(f"Entering sub-configuration for {field_name}:")
            config_data[field_name] = prompt_for_config(nested_type, existing_value)
        elif (
            inspect.isclass(field_type)
            and issubclass(field_type, BaseModel)
            and len(field_type.__fields__) > 0
        ):
            print(f"\nEntering sub-configuration for {field_name}:")
            config_data[field_name] = prompt_for_config(
                field_type,
                existing_value,
            )
        else:
            prompt = f"Enter value for {field_name}"
            if existing_value is not None:
                prompt += f" (existing: {existing_value})"
            elif default_value is not None:
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
                    elif is_optional(field_type) or not is_required:
                        config_data[field_name] = None
                        break
                    else:
                        print("This field is required. Please provide a value.")
                        continue
                else:
                    try:
                        # Handle Optional types
                        if is_optional(field_type):
                            if user_input.lower() == "none":
                                value = None
                            else:
                                field_type = get_non_none_type(field_type)
                                value = user_input

                        # Handle List of primitives
                        elif is_list_of_primitives(field_type):
                            try:
                                value = json.loads(user_input)
                                if not isinstance(value, list):
                                    raise ValueError(
                                        "Input must be a JSON-encoded list"
                                    )
                                element_type = get_args(field_type)[0]
                                value = [element_type(item) for item in value]

                            except json.JSONDecodeError:
                                print(
                                    "Invalid JSON. Please enter a valid JSON-encoded list."
                                )
                                continue
                            except ValueError as e:
                                print(f"{str(e)}")
                                continue

                        # Convert the input to the correct type
                        elif inspect.isclass(field_type) and issubclass(
                            field_type, BaseModel
                        ):
                            # For nested BaseModels, we assume a dictionary-like string input
                            import ast

                            value = field_type(**ast.literal_eval(user_input))
                        else:
                            value = field_type(user_input)

                    except ValueError:
                        print(
                            f"Invalid input. Expected type: {getattr(field_type, '__name__', str(field_type))}"
                        )
                        continue

                try:
                    # Validate the field using our manual validation function
                    validated_value = manually_validate_field(
                        config_type, field_name, value
                    )
                    config_data[field_name] = validated_value
                    break
                except ValueError as e:
                    print(f"Validation error: {str(e)}")

    return config_type(**config_data)
