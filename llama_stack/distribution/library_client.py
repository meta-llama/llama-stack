# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import inspect
from pathlib import Path
from typing import Any, get_args, get_origin, Optional

import yaml
from llama_stack_client import LlamaStackClient, NOT_GIVEN
from pydantic import TypeAdapter
from rich.console import Console

from termcolor import cprint

from llama_stack.distribution.build import print_pip_install_help
from llama_stack.distribution.configure import parse_and_maybe_upgrade_config
from llama_stack.distribution.resolver import ProviderRegistry
from llama_stack.distribution.server.endpoints import get_all_api_endpoints
from llama_stack.distribution.stack import (
    construct_stack,
    get_stack_run_config_from_template,
    replace_env_vars,
)


class LlamaStackAsLibraryClient(LlamaStackClient):
    def __init__(
        self,
        config_path_or_template_name: str,
        custom_provider_registry: Optional[ProviderRegistry] = None,
    ):
        if config_path_or_template_name.endswith(".yaml"):
            config_path = Path(config_path_or_template_name)
            if not config_path.exists():
                raise ValueError(f"Config file {config_path} does not exist")
            config_dict = replace_env_vars(yaml.safe_load(config_path.read_text()))
            config = parse_and_maybe_upgrade_config(config_dict)
        else:
            # template
            config = get_stack_run_config_from_template(config_path_or_template_name)

        self.config_path_or_template_name = config_path_or_template_name
        self.config = config
        self.custom_provider_registry = custom_provider_registry

        super().__init__()

    async def initialize(self):
        try:
            self.impls = await construct_stack(
                self.config, self.custom_provider_registry
            )
        except ModuleNotFoundError as e:
            cprint(
                "Using llama-stack as a library requires installing dependencies depending on the template (providers) you choose.\n",
                "yellow",
            )
            print_pip_install_help(self.config.providers)
            raise e

        console = Console()
        console.print(f"Using config [blue]{self.config_path_or_template_name}[/blue]:")
        console.print(yaml.dump(self.config.model_dump(), indent=2))

        endpoints = get_all_api_endpoints()
        endpoint_impls = {}
        for api, api_endpoints in endpoints.items():
            for endpoint in api_endpoints:
                impl = self.impls[api]
                func = getattr(impl, endpoint.name)
                endpoint_impls[endpoint.route] = func

        self.endpoint_impls = endpoint_impls

    async def get(
        self,
        path: str,
        *,
        stream=False,
        **kwargs,
    ):
        if not self.endpoint_impls:
            raise ValueError("Client not initialized")

        if stream:
            return self._call_streaming(path, "GET")
        else:
            return await self._call_non_streaming(path, "GET")

    async def post(
        self,
        path: str,
        *,
        body: dict = None,
        stream=False,
        **kwargs,
    ):
        if not self.endpoint_impls:
            raise ValueError("Client not initialized")

        if stream:
            return self._call_streaming(path, "POST", body)
        else:
            return await self._call_non_streaming(path, "POST", body)

    async def _call_non_streaming(self, path: str, method: str, body: dict = None):
        func = self.endpoint_impls.get(path)
        if not func:
            raise ValueError(f"No endpoint found for {path}")

        body = self._convert_body(path, body)
        return await func(**body)

    async def _call_streaming(self, path: str, method: str, body: dict = None):
        func = self.endpoint_impls.get(path)
        if not func:
            raise ValueError(f"No endpoint found for {path}")

        body = self._convert_body(path, body)
        async for chunk in await func(**body):
            yield chunk

    def _convert_body(self, path: str, body: Optional[dict] = None) -> dict:
        if not body:
            return {}

        func = self.endpoint_impls[path]
        sig = inspect.signature(func)

        # Strip NOT_GIVENs to use the defaults in signature
        body = {k: v for k, v in body.items() if v is not NOT_GIVEN}

        # Convert parameters to Pydantic models where needed
        converted_body = {}
        for param_name, param in sig.parameters.items():
            if param_name in body:
                value = body.get(param_name)
                converted_body[param_name] = self._convert_param(
                    param.annotation, value
                )
        return converted_body

    def _convert_param(self, annotation: Any, value: Any) -> Any:
        if isinstance(annotation, type) and annotation in {str, int, float, bool}:
            return value

        origin = get_origin(annotation)
        if origin is list:
            item_type = get_args(annotation)[0]
            try:
                return [self._convert_param(item_type, item) for item in value]
            except Exception:
                return value

        elif origin is dict:
            key_type, val_type = get_args(annotation)
            try:
                return {k: self._convert_param(val_type, v) for k, v in value.items()}
            except Exception:
                return value

        try:
            # Handle Pydantic models and discriminated unions
            return TypeAdapter(annotation).validate_python(value)
        except Exception as e:
            cprint(
                f"Warning: direct client failed to convert parameter {value} into {annotation}: {e}",
                "yellow",
            )
            return value
