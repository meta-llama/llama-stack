# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import re
from typing import Any

from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="core")


class EnvVarError(Exception):
    def __init__(self, var_name: str, path: str = ""):
        self.var_name = var_name
        self.path = path
        super().__init__(f"Environment variable '{var_name}' not set or empty{f' at {path}' if path else ''}")


def replace_env_vars(config: Any, path: str = "") -> Any:
    if isinstance(config, dict):
        result_dict: dict[Any, Any] = {}
        for k, v in config.items():
            try:
                result_dict[k] = replace_env_vars(v, f"{path}.{k}" if path else k)
            except EnvVarError as e:
                raise EnvVarError(e.var_name, e.path) from None
        return result_dict

    elif isinstance(config, list):
        result_list: list[Any] = []
        for i, v in enumerate(config):
            try:
                result_list.append(replace_env_vars(v, f"{path}[{i}]"))
            except EnvVarError as e:
                raise EnvVarError(e.var_name, e.path) from None
        return result_list

    elif isinstance(config, str):
        # Pattern supports bash-like syntax: := for default and :+ for conditional and a optional value
        pattern = r"\${env\.([A-Z0-9_]+)(?::([=+])([^}]*))?}"

        def get_env_var(match: re.Match):
            env_var = match.group(1)
            operator = match.group(2)  # '=' for default, '+' for conditional
            value_expr = match.group(3)

            env_value = os.environ.get(env_var)

            if operator == "=":  # Default value syntax: ${env.FOO:=default}
                if not env_value:
                    # value_expr returns empty string (not None) when not matched
                    # This means ${env.FOO:=} is an error
                    if value_expr == "":
                        raise EnvVarError(env_var, path)
                    else:
                        value = value_expr
                else:
                    value = env_value
            elif operator == "+":  # Conditional value syntax: ${env.FOO:+value_if_set}
                if env_value:
                    value = value_expr
                else:
                    # If env var is not set, return empty string for the conditional case
                    value = ""
            else:  # No operator case: ${env.FOO}
                if not env_value:
                    raise EnvVarError(env_var, path)
                value = env_value

            # expand "~" from the values
            return os.path.expanduser(value)

        try:
            result: str = re.sub(pattern, get_env_var, config)
            return _convert_string_to_proper_type(result)
        except EnvVarError as e:
            raise EnvVarError(e.var_name, e.path) from None

    return config


def validate_env_pair(env_pair: str) -> tuple[str, str]:
    """Validate and split an environment variable key-value pair."""
    try:
        key, value = env_pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in environment variable pair: {env_pair}")
        if not all(c.isalnum() or c == "_" for c in key):
            raise ValueError(f"Key must contain only alphanumeric characters and underscores: {key}")
        return key, value
    except ValueError as e:
        raise ValueError(
            f"Invalid environment variable format '{env_pair}': {str(e)}. Expected format: KEY=value"
        ) from e


def _convert_string_to_proper_type(value: str) -> Any:
    # This might be tricky depending on what the config type is, if  'str | None' we are
    # good, if 'str' we need to keep the empty string... 'str | None' is more common and
    # providers config should be typed this way.
    # TODO: we could try to load the config class and see if the config has a field with type 'str | None'
    # and then convert the empty string to None or not
    if value == "":
        return None

    lowered = value.lower()
    if lowered == "true":
        return True
    elif lowered == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value
