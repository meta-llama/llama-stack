# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import inspect
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin

import httpx
import yaml
from fastapi import Response as FastAPIResponse
from llama_stack_client import (
    NOT_GIVEN,
    APIResponse,
    AsyncAPIResponse,
    AsyncLlamaStackClient,
    AsyncStream,
    LlamaStackClient,
)
from pydantic import BaseModel, TypeAdapter
from rich.console import Console
from termcolor import cprint

from llama_stack.distribution.build import print_pip_install_help
from llama_stack.distribution.configure import parse_and_maybe_upgrade_config
from llama_stack.distribution.datatypes import Api, BuildConfig, BuildProvider, DistributionSpec
from llama_stack.distribution.request_headers import (
    PROVIDER_DATA_VAR,
    request_provider_data_context,
)
from llama_stack.distribution.resolver import ProviderRegistry
from llama_stack.distribution.server.routes import find_matching_route, initialize_route_impls
from llama_stack.distribution.stack import (
    construct_stack,
    get_stack_run_config_from_template,
    replace_env_vars,
)
from llama_stack.distribution.utils.config import redact_sensitive_fields
from llama_stack.distribution.utils.context import preserve_contexts_async_generator
from llama_stack.distribution.utils.exec import in_notebook
from llama_stack.providers.utils.telemetry.tracing import (
    CURRENT_TRACE_CONTEXT,
    end_trace,
    setup_logger,
    start_trace,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
            logger.error(f"Error converting list {value} into {item_type}")
            return value

    elif origin is dict:
        key_type, val_type = get_args(annotation)
        try:
            return {k: convert_to_pydantic(val_type, v) for k, v in value.items()}
        except Exception:
            logger.error(f"Error converting dict {value} into {val_type}")
            return value

    try:
        # Handle Pydantic models and discriminated unions
        return TypeAdapter(annotation).validate_python(value)

    except Exception as e:
        # TODO: this is workardound for having Union[str, AgentToolGroup] in API schema.
        # We should get rid of any non-discriminated unions in the API schema.
        if origin is Union:
            for union_type in get_args(annotation):
                try:
                    return convert_to_pydantic(union_type, value)
                except Exception:
                    continue
            logger.warning(
                f"Warning: direct client failed to convert parameter {value} into {annotation}: {e}",
            )
        raise ValueError(f"Failed to convert parameter {value} into {annotation}: {e}") from e


class LibraryClientUploadFile:
    """LibraryClient UploadFile object that mimics FastAPI's UploadFile interface."""

    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self.content = content
        self.content_type = "application/octet-stream"

    async def read(self) -> bytes:
        return self.content


class LibraryClientHttpxResponse:
    """LibraryClient httpx Response object for FastAPI Response conversion."""

    def __init__(self, response):
        self.content = response.body if isinstance(response.body, bytes) else response.body.encode()
        self.status_code = response.status_code
        self.headers = response.headers


class LlamaStackAsLibraryClient(LlamaStackClient):
    def __init__(
        self,
        config_path_or_template_name: str,
        skip_logger_removal: bool = False,
        custom_provider_registry: ProviderRegistry | None = None,
        provider_data: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.async_client = AsyncLlamaStackAsLibraryClient(
            config_path_or_template_name, custom_provider_registry, provider_data
        )
        self.pool_executor = ThreadPoolExecutor(max_workers=4)
        self.skip_logger_removal = skip_logger_removal
        self.provider_data = provider_data

        self.loop = asyncio.new_event_loop()

    def initialize(self):
        if in_notebook():
            import nest_asyncio

            nest_asyncio.apply()
            if not self.skip_logger_removal:
                self._remove_root_logger_handlers()

        # use a new event loop to avoid interfering with the main event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.async_client.initialize())
        finally:
            asyncio.set_event_loop(None)

    def _remove_root_logger_handlers(self):
        """
        Remove all handlers from the root logger. Needed to avoid polluting the console with logs.
        """
        root_logger = logging.getLogger()

        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            logger.info(f"Removed handler {handler.__class__.__name__} from root logger")

    def request(self, *args, **kwargs):
        loop = self.loop
        asyncio.set_event_loop(loop)

        if kwargs.get("stream"):

            def sync_generator():
                try:
                    async_stream = loop.run_until_complete(self.async_client.request(*args, **kwargs))
                    while True:
                        chunk = loop.run_until_complete(async_stream.__anext__())
                        yield chunk
                except StopAsyncIteration:
                    pass
                finally:
                    pending = asyncio.all_tasks(loop)
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

            return sync_generator()
        else:
            try:
                result = loop.run_until_complete(self.async_client.request(*args, **kwargs))
            finally:
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            return result


class AsyncLlamaStackAsLibraryClient(AsyncLlamaStackClient):
    def __init__(
        self,
        config_path_or_template_name: str,
        custom_provider_registry: ProviderRegistry | None = None,
        provider_data: dict[str, Any] | None = None,
    ):
        super().__init__()
        # when using the library client, we should not log to console since many
        # of our logs are intended for server-side usage
        current_sinks = os.environ.get("TELEMETRY_SINKS", "sqlite").split(",")
        os.environ["TELEMETRY_SINKS"] = ",".join(sink for sink in current_sinks if sink != "console")

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
        self.provider_data = provider_data

    async def initialize(self) -> bool:
        try:
            self.route_impls = None
            self.impls = await construct_stack(self.config, self.custom_provider_registry)
        except ModuleNotFoundError as _e:
            cprint(_e.msg, color="red", file=sys.stderr)
            cprint(
                "Using llama-stack as a library requires installing dependencies depending on the template (providers) you choose.\n",
                color="yellow",
                file=sys.stderr,
            )
            if self.config_path_or_template_name.endswith(".yaml"):
                providers: dict[str, list[BuildProvider]] = {}
                for api, run_providers in self.config.providers.items():
                    for provider in run_providers:
                        providers.setdefault(api, []).append(
                            BuildProvider(provider_type=provider.provider_type, module=provider.module)
                        )
                providers = dict(providers)
                build_config = BuildConfig(
                    distribution_spec=DistributionSpec(
                        providers=providers,
                    ),
                    external_providers_dir=self.config.external_providers_dir,
                )
                print_pip_install_help(build_config)
            else:
                prefix = "!" if in_notebook() else ""
                cprint(
                    f"Please run:\n\n{prefix}llama stack build --template {self.config_path_or_template_name} --image-type venv\n\n",
                    "yellow",
                    file=sys.stderr,
                )
            cprint(
                "Please check your internet connection and try again.",
                "red",
                file=sys.stderr,
            )
            raise _e

        if Api.telemetry in self.impls:
            setup_logger(self.impls[Api.telemetry])

        if not os.environ.get("PYTEST_CURRENT_TEST"):
            console = Console()
            console.print(f"Using config [blue]{self.config_path_or_template_name}[/blue]:")
            safe_config = redact_sensitive_fields(self.config.model_dump())
            console.print(yaml.dump(safe_config, indent=2))

        self.route_impls = initialize_route_impls(self.impls)
        return True

    async def request(
        self,
        cast_to: Any,
        options: Any,
        *,
        stream=False,
        stream_cls=None,
    ):
        if not self.route_impls:
            raise ValueError("Client not initialized")

        # Create headers with provider data if available
        headers = options.headers or {}
        if self.provider_data:
            keys = ["X-LlamaStack-Provider-Data", "x-llamastack-provider-data"]
            if all(key not in headers for key in keys):
                headers["X-LlamaStack-Provider-Data"] = json.dumps(self.provider_data)

        # Use context manager for provider data
        with request_provider_data_context(headers):
            if stream:
                response = await self._call_streaming(
                    cast_to=cast_to,
                    options=options,
                    stream_cls=stream_cls,
                )
            else:
                response = await self._call_non_streaming(
                    cast_to=cast_to,
                    options=options,
                )
            return response

    def _handle_file_uploads(self, options: Any, body: dict) -> tuple[dict, list[str]]:
        """Handle file uploads from OpenAI client and add them to the request body."""
        if not (hasattr(options, "files") and options.files):
            return body, []

        if not isinstance(options.files, list):
            return body, []

        field_names = []
        for file_tuple in options.files:
            if not (isinstance(file_tuple, tuple) and len(file_tuple) >= 2):
                continue

            field_name = file_tuple[0]
            file_object = file_tuple[1]

            if isinstance(file_object, BytesIO):
                file_object.seek(0)
                file_content = file_object.read()
                filename = getattr(file_object, "name", "uploaded_file")
                field_names.append(field_name)
                body[field_name] = LibraryClientUploadFile(filename, file_content)

        return body, field_names

    async def _call_non_streaming(
        self,
        *,
        cast_to: Any,
        options: Any,
    ):
        if self.route_impls is None:
            raise ValueError("Client not initialized")

        path = options.url
        body = options.params or {}
        body |= options.json_data or {}

        matched_func, path_params, route_path, webmethod = find_matching_route(options.method, path, self.route_impls)
        body |= path_params

        body, field_names = self._handle_file_uploads(options, body)

        body = self._convert_body(path, options.method, body, exclude_params=set(field_names))

        trace_path = webmethod.descriptive_name or route_path
        await start_trace(trace_path, {"__location__": "library_client"})
        try:
            result = await matched_func(**body)
        finally:
            await end_trace()

        # Handle FastAPI Response objects (e.g., from file content retrieval)
        if isinstance(result, FastAPIResponse):
            return LibraryClientHttpxResponse(result)

        json_content = json.dumps(convert_pydantic_to_json_value(result))

        filtered_body = {k: v for k, v in body.items() if not isinstance(v, LibraryClientUploadFile)}
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
                headers=options.headers or {},
                json=convert_pydantic_to_json_value(filtered_body),
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

    async def _call_streaming(
        self,
        *,
        cast_to: Any,
        options: Any,
        stream_cls: Any,
    ):
        if self.route_impls is None:
            raise ValueError("Client not initialized")

        path = options.url
        body = options.params or {}
        body |= options.json_data or {}
        func, path_params, route_path, webmethod = find_matching_route(options.method, path, self.route_impls)
        body |= path_params

        body = self._convert_body(path, options.method, body)

        trace_path = webmethod.descriptive_name or route_path
        await start_trace(trace_path, {"__location__": "library_client"})

        async def gen():
            try:
                async for chunk in await func(**body):
                    data = json.dumps(convert_pydantic_to_json_value(chunk))
                    sse_event = f"data: {data}\n\n"
                    yield sse_event.encode("utf-8")
            finally:
                await end_trace()

        wrapped_gen = preserve_contexts_async_generator(gen(), [CURRENT_TRACE_CONTEXT, PROVIDER_DATA_VAR])

        mock_response = httpx.Response(
            status_code=httpx.codes.OK,
            content=wrapped_gen,
            headers={
                "Content-Type": "application/json",
            },
            request=httpx.Request(
                method=options.method,
                url=options.url,
                params=options.params,
                headers=options.headers or {},
                json=convert_pydantic_to_json_value(body),
            ),
        )

        # we use asynchronous impl always internally and channel all requests to AsyncLlamaStackClient
        # however, the top-level caller may be a SyncAPIClient -- so its stream_cls might be a Stream (SyncStream)
        # so we need to convert it to AsyncStream
        # mypy can't track runtime variables inside the [...] of a generic, so ignore that check
        args = get_args(stream_cls)
        stream_cls = AsyncStream[args[0]]  # type: ignore[valid-type]
        response = AsyncAPIResponse(
            raw=mock_response,
            client=self,
            cast_to=cast_to,
            options=options,
            stream=True,
            stream_cls=stream_cls,
        )
        return await response.parse()

    def _convert_body(
        self, path: str, method: str, body: dict | None = None, exclude_params: set[str] | None = None
    ) -> dict:
        if not body:
            return {}

        if self.route_impls is None:
            raise ValueError("Client not initialized")

        exclude_params = exclude_params or set()

        func, _, _, _ = find_matching_route(method, path, self.route_impls)
        sig = inspect.signature(func)

        # Strip NOT_GIVENs to use the defaults in signature
        body = {k: v for k, v in body.items() if v is not NOT_GIVEN}

        # Convert parameters to Pydantic models where needed
        converted_body = {}
        for param_name, param in sig.parameters.items():
            if param_name in body:
                value = body.get(param_name)
                if param_name in exclude_params:
                    converted_body[param_name] = value
                else:
                    converted_body[param_name] = convert_to_pydantic(param.annotation, value)

        return converted_body
