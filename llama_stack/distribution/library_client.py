# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import inspect
import json
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Any, Generator, get_args, get_origin, Optional, TypeVar

import httpx

import yaml
from llama_stack_client import (
    APIResponse,
    AsyncAPIResponse,
    AsyncLlamaStackClient,
    AsyncStream,
    LlamaStackClient,
    NOT_GIVEN,
)
from pydantic import BaseModel, TypeAdapter
from rich.console import Console

from termcolor import cprint

from llama_stack.distribution.build import print_pip_install_help
from llama_stack.distribution.configure import parse_and_maybe_upgrade_config
from llama_stack.distribution.datatypes import Api
from llama_stack.distribution.resolver import ProviderRegistry
from llama_stack.distribution.server.endpoints import get_all_api_endpoints
from llama_stack.distribution.stack import (
    construct_stack,
    get_stack_run_config_from_template,
    replace_env_vars,
)

from llama_stack.providers.utils.telemetry.tracing import (
    end_trace,
    setup_logger,
    start_trace,
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
            async for item in await gen:
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


def convert_pydantic_to_json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    elif isinstance(value, list):
        return [convert_pydantic_to_json_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: convert_pydantic_to_json_value(v) for k, v in value.items()}
    elif isinstance(value, BaseModel):
        return json.loads(value.model_dump_json())
    else:
        return value


def convert_to_pydantic(annotation: Any, value: Any) -> Any:
    if isinstance(annotation, type) and annotation in {str, int, float, bool}:
        return value

    origin = get_origin(annotation)
    if origin is list:
        item_type = get_args(annotation)[0]
        try:
            return [convert_to_pydantic(item_type, item) for item in value]
        except Exception:
            print(f"Error converting list {value}")
            return value

    elif origin is dict:
        key_type, val_type = get_args(annotation)
        try:
            return {k: convert_to_pydantic(val_type, v) for k, v in value.items()}
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

    def request(self, *args, **kwargs):
        if kwargs.get("stream"):
            return stream_across_asyncio_run_boundary(
                lambda: self.async_client.request(*args, **kwargs),
                self.pool_executor,
            )
        else:
            return asyncio.run(self.async_client.request(*args, **kwargs))


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
                prefix = "!" if in_notebook() else ""
                cprint(
                    f"Please run:\n\n{prefix}llama stack build --template {self.config_path_or_template_name} --image-type venv\n\n",
                    "yellow",
                )
            return False

        if Api.telemetry in self.impls:
            setup_logger(self.impls[Api.telemetry])

        console = Console()
        console.print(f"Using config [blue]{self.config_path_or_template_name}[/blue]:")
        console.print(yaml.dump(self.config.model_dump(), indent=2))

        endpoints = get_all_api_endpoints()
        endpoint_impls = {}
        for api, api_endpoints in endpoints.items():
            if api not in self.impls:
                continue
            for endpoint in api_endpoints:
                impl = self.impls[api]
                func = getattr(impl, endpoint.name)
                endpoint_impls[endpoint.route] = func

        self.endpoint_impls = endpoint_impls
        return True

    async def request(
        self,
        cast_to: Any,
        options: Any,
        *,
        stream=False,
        stream_cls=None,
    ):
        if not self.endpoint_impls:
            raise ValueError("Client not initialized")

        if stream:
            return self._call_streaming(
                cast_to=cast_to,
                options=options,
                stream_cls=stream_cls,
            )
        else:
            return await self._call_non_streaming(
                cast_to=cast_to,
                options=options,
            )

    async def _call_non_streaming(
        self,
        *,
        cast_to: Any,
        options: Any,
    ):
        path = options.url

        body = options.params or {}
        body |= options.json_data or {}
        await start_trace(path, {"__location__": "library_client"})
        try:
            func = self.endpoint_impls.get(path)
            if not func:
                raise ValueError(f"No endpoint found for {path}")

            body = self._convert_body(path, body)
            result = await func(**body)

            json_content = json.dumps(convert_pydantic_to_json_value(result))
            mock_response = httpx.Response(
                status_code=httpx.codes.OK,
                content=json_content.encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                },
                request=httpx.Request(
                    method=options.method,
                    url=options.url,
                    params=options.params,
                    headers=options.headers,
                    json=options.json_data,
                ),
            )
            response = APIResponse(
                raw=mock_response,
                client=self,
                cast_to=cast_to,
                options=options,
                stream=False,
                stream_cls=None,
            )
            return response.parse()
        finally:
            await end_trace()

    async def _call_streaming(
        self,
        *,
        cast_to: Any,
        options: Any,
        stream_cls: Any,
    ):
        path = options.url
        body = options.params or {}
        body |= options.json_data or {}
        await start_trace(path, {"__location__": "library_client"})
        try:
            func = self.endpoint_impls.get(path)
            if not func:
                raise ValueError(f"No endpoint found for {path}")

            body = self._convert_body(path, body)

            async def gen():
                async for chunk in await func(**body):
                    data = json.dumps(convert_pydantic_to_json_value(chunk))
                    sse_event = f"data: {data}\n\n"
                    yield sse_event.encode("utf-8")

            mock_response = httpx.Response(
                status_code=httpx.codes.OK,
                content=gen(),
                headers={
                    "Content-Type": "application/json",
                },
                request=httpx.Request(
                    method=options.method,
                    url=options.url,
                    params=options.params,
                    headers=options.headers,
                    json=options.json_data,
                ),
            )

            # we use asynchronous impl always internally and channel all requests to AsyncLlamaStackClient
            # however, the top-level caller may be a SyncAPIClient -- so its stream_cls might be a Stream (SyncStream)
            # so we need to convert it to AsyncStream
            args = get_args(stream_cls)
            stream_cls = AsyncStream[args[0]]
            response = AsyncAPIResponse(
                raw=mock_response,
                client=self,
                cast_to=cast_to,
                options=options,
                stream=True,
                stream_cls=stream_cls,
            )
            return await response.parse()
        finally:
            await end_trace()

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
                converted_body[param_name] = convert_to_pydantic(
                    param.annotation, value
                )
        return converted_body
