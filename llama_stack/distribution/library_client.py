# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import inspect
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Generator, get_args, get_origin, Optional, TypeVar

import yaml
from llama_stack_client import AsyncLlamaStackClient, LlamaStackClient, NOT_GIVEN
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

T = TypeVar("T")


def in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def stream_across_asyncio_run_boundary(
    async_gen_maker,
    pool_executor: ThreadPoolExecutor,
) -> Generator[T, None, None]:
    result_queue = queue.Queue()
    stop_event = threading.Event()

    async def consumer():
        # make sure we make the generator in the event loop context
        gen = await async_gen_maker()
        try:
            async for item in gen:
                result_queue.put(item)
        except Exception as e:
            print(f"Error in generator {e}")
            result_queue.put(e)
        except asyncio.CancelledError:
            return
        finally:
            result_queue.put(StopIteration)
            stop_event.set()

    def run_async():
        # Run our own loop to avoid double async generator cleanup which is done
        # by asyncio.run()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            task = loop.create_task(consumer())
            loop.run_until_complete(task)
        finally:
            # Handle pending tasks like a generator's athrow()
            pending = asyncio.all_tasks(loop)
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.close()

    future = pool_executor.submit(run_async)

    try:
        # yield results as they come in
        while not stop_event.is_set() or not result_queue.empty():
            try:
                item = result_queue.get(timeout=0.1)
                if item is StopIteration:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
            except queue.Empty:
                continue
    finally:
        future.result()


class LlamaStackAsLibraryClient(LlamaStackClient):
    def __init__(
        self,
        config_path_or_template_name: str,
        custom_provider_registry: Optional[ProviderRegistry] = None,
    ):
        super().__init__()
        self.async_client = AsyncLlamaStackAsLibraryClient(
            config_path_or_template_name, custom_provider_registry
        )
        self.pool_executor = ThreadPoolExecutor(max_workers=4)

    def initialize(self):
        if in_notebook():
            import nest_asyncio

            nest_asyncio.apply()

        return asyncio.run(self.async_client.initialize())

    def get(self, *args, **kwargs):
        if kwargs.get("stream"):
            return stream_across_asyncio_run_boundary(
                lambda: self.async_client.get(*args, **kwargs),
                self.pool_executor,
            )
        else:
            return asyncio.run(self.async_client.get(*args, **kwargs))

    def post(self, *args, **kwargs):
        if kwargs.get("stream"):
            return stream_across_asyncio_run_boundary(
                lambda: self.async_client.post(*args, **kwargs),
                self.pool_executor,
            )
        else:
            return asyncio.run(self.async_client.post(*args, **kwargs))


class AsyncLlamaStackAsLibraryClient(AsyncLlamaStackClient):
    def __init__(
        self,
        config_path_or_template_name: str,
        custom_provider_registry: Optional[ProviderRegistry] = None,
    ):
        super().__init__()

        # when using the library client, we should not log to console since many
        # of our logs are intended for server-side usage
        os.environ["TELEMETRY_SINKS"] = "sqlite"

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

    async def initialize(self):
        try:
            self.impls = await construct_stack(
                self.config, self.custom_provider_registry
            )
        except ModuleNotFoundError as _e:
            cprint(
                "Using llama-stack as a library requires installing dependencies depending on the template (providers) you choose.\n",
                "yellow",
            )
            if self.config_path_or_template_name.endswith(".yaml"):
                print_pip_install_help(self.config.providers)
            else:
                cprint(
                    f"Please run:\n\nllama stack build --template {self.config_path_or_template_name} --image-type venv\n\n",
                    "yellow",
                )
            return False

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
        return True

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
                print(f"Error converting list {value}")
                return value

        elif origin is dict:
            key_type, val_type = get_args(annotation)
            try:
                return {k: self._convert_param(val_type, v) for k, v in value.items()}
            except Exception:
                print(f"Error converting dict {value}")
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
