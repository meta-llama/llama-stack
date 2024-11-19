# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import asyncio
import functools
import inspect
import json
import os
import signal
import sys
import traceback
import warnings

from contextlib import asynccontextmanager
from ssl import SSLError
from typing import Any, Dict, Optional

import httpx
import yaml

from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ValidationError
from termcolor import cprint
from typing_extensions import Annotated

from llama_stack.distribution.distribution import builtin_automatically_routed_apis

from llama_stack.providers.utils.telemetry.tracing import (
    end_trace,
    setup_logger,
    SpanStatus,
    start_trace,
)
from llama_stack.distribution.datatypes import *  # noqa: F403
from llama_stack.distribution.request_headers import set_request_provider_data
from llama_stack.distribution.resolver import InvalidProviderError
from llama_stack.distribution.stack import (
    construct_stack,
    replace_env_vars,
    validate_env_pair,
)

from .endpoints import get_all_api_endpoints


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


if os.environ.get("LLAMA_STACK_TRACE_WARNINGS"):
    warnings.showwarning = warn_with_traceback


def create_sse_event(data: Any) -> str:
    if isinstance(data, BaseModel):
        data = data.model_dump_json()
    else:
        data = json.dumps(data)

    return f"data: {data}\n\n"


async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exception(exc)
    http_exc = translate_exception(exc)

    return JSONResponse(
        status_code=http_exc.status_code, content={"error": {"detail": http_exc.detail}}
    )


def translate_exception(exc: Exception) -> Union[HTTPException, RequestValidationError]:
    if isinstance(exc, ValidationError):
        exc = RequestValidationError(exc.raw_errors)

    if isinstance(exc, RequestValidationError):
        return HTTPException(
            status_code=400,
            detail={
                "errors": [
                    {
                        "loc": list(error["loc"]),
                        "msg": error["msg"],
                        "type": error["type"],
                    }
                    for error in exc.errors()
                ]
            },
        )
    elif isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=f"Invalid value: {str(exc)}")
    elif isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail=f"Permission denied: {str(exc)}")
    elif isinstance(exc, TimeoutError):
        return HTTPException(status_code=504, detail=f"Operation timed out: {str(exc)}")
    elif isinstance(exc, NotImplementedError):
        return HTTPException(status_code=501, detail=f"Not implemented: {str(exc)}")
    else:
        return HTTPException(
            status_code=500,
            detail="Internal server error: An unexpected error occurred.",
        )


async def passthrough(
    request: Request,
    downstream_url: str,
    downstream_headers: Optional[Dict[str, str]] = None,
):
    await start_trace(request.path, {"downstream_url": downstream_url})

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.update(downstream_headers or {})

    content = await request.body()

    client = httpx.AsyncClient()
    erred = False
    try:
        req = client.build_request(
            method=request.method,
            url=downstream_url,
            headers=headers,
            content=content,
            params=request.query_params,
        )
        response = await client.send(req, stream=True)

        async def stream_response():
            async for chunk in response.aiter_raw(chunk_size=64):
                yield chunk

            await response.aclose()
            await client.aclose()

        return StreamingResponse(
            stream_response(),
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.headers.get("content-type"),
        )

    except httpx.ReadTimeout:
        erred = True
        return Response(content="Downstream server timed out", status_code=504)
    except httpx.NetworkError as e:
        erred = True
        return Response(content=f"Network error: {str(e)}", status_code=502)
    except httpx.TooManyRedirects:
        erred = True
        return Response(content="Too many redirects", status_code=502)
    except SSLError as e:
        erred = True
        return Response(content=f"SSL error: {str(e)}", status_code=502)
    except httpx.HTTPStatusError as e:
        erred = True
        return Response(content=str(e), status_code=e.response.status_code)
    except Exception as e:
        erred = True
        return Response(content=f"Unexpected error: {str(e)}", status_code=500)
    finally:
        await end_trace(SpanStatus.OK if not erred else SpanStatus.ERROR)


def handle_sigint(app, *args, **kwargs):
    print("SIGINT or CTRL-C detected. Exiting gracefully...")

    async def run_shutdown():
        for impl in app.__llama_stack_impls__.values():
            print(f"Shutting down {impl}")
            await impl.shutdown()

    asyncio.run(run_shutdown())

    loop = asyncio.get_event_loop()
    for task in asyncio.all_tasks(loop):
        task.cancel()

    loop.stop()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up")
    yield

    print("Shutting down")
    for impl in app.__llama_stack_impls__.values():
        await impl.shutdown()


def is_streaming_request(func_name: str, request: Request, **kwargs):
    # TODO: pass the api method and punt it to the Protocol definition directly
    return kwargs.get("stream", False)


async def maybe_await(value):
    if inspect.iscoroutine(value):
        return await value
    return value


async def sse_generator(event_gen):
    try:
        event_gen = await event_gen
        async for item in event_gen:
            yield create_sse_event(item)
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        print("Generator cancelled")
        await event_gen.aclose()
    except Exception as e:
        traceback.print_exception(e)
        yield create_sse_event(
            {
                "error": {
                    "message": str(translate_exception(e)),
                },
            }
        )
    finally:
        await end_trace()


def create_dynamic_typed_route(func: Any, method: str):
    async def endpoint(request: Request, **kwargs):
        await start_trace(func.__name__)

        set_request_provider_data(request.headers)

        is_streaming = is_streaming_request(func.__name__, request, **kwargs)
        try:
            if is_streaming:
                return StreamingResponse(
                    sse_generator(func(**kwargs)), media_type="text/event-stream"
                )
            else:
                value = func(**kwargs)
                return await maybe_await(value)
        except Exception as e:
            traceback.print_exception(e)
            raise translate_exception(e) from e
        finally:
            await end_trace()

    sig = inspect.signature(func)
    new_params = [
        inspect.Parameter(
            "request", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Request
        )
    ]
    new_params.extend(sig.parameters.values())

    if method == "post":
        # make sure every parameter is annotated with Body() so FASTAPI doesn't
        # do anything too intelligent and ask for some parameters in the query
        # and some in the body
        new_params = [new_params[0]] + [
            param.replace(annotation=Annotated[param.annotation, Body(..., embed=True)])
            for param in new_params[1:]
        ]

    endpoint.__signature__ = sig.replace(parameters=new_params)

    return endpoint


def main():
    """Start the LlamaStack server."""
    parser = argparse.ArgumentParser(description="Start the LlamaStack server.")
    parser.add_argument(
        "--yaml-config",
        default="llamastack-run.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument(
        "--disable-ipv6", action="store_true", help="Whether to disable IPv6 support"
    )
    parser.add_argument(
        "--env",
        action="append",
        help="Environment variables in KEY=value format. Can be specified multiple times.",
    )

    args = parser.parse_args()
    if args.env:
        for env_pair in args.env:
            try:
                key, value = validate_env_pair(env_pair)
                print(f"Setting CLI environment variable {key} => {value}")
                os.environ[key] = value
            except ValueError as e:
                print(f"Error: {str(e)}")
                sys.exit(1)

    with open(args.yaml_config, "r") as fp:
        config = replace_env_vars(yaml.safe_load(fp))
        config = StackRunConfig(**config)

    app = FastAPI()

    try:
        impls = asyncio.run(construct_stack(config))
    except InvalidProviderError:
        sys.exit(1)

    if Api.telemetry in impls:
        setup_logger(impls[Api.telemetry])

    all_endpoints = get_all_api_endpoints()

    if config.apis:
        apis_to_serve = set(config.apis)
    else:
        apis_to_serve = set(impls.keys())

    for inf in builtin_automatically_routed_apis():
        # if we do not serve the corresponding router API, we should not serve the routing table API
        if inf.router_api.value not in apis_to_serve:
            continue
        apis_to_serve.add(inf.routing_table_api.value)

    apis_to_serve.add("inspect")
    for api_str in apis_to_serve:
        api = Api(api_str)

        endpoints = all_endpoints[api]
        impl = impls[api]

        for endpoint in endpoints:
            if not hasattr(impl, endpoint.name):
                # ideally this should be a typing violation already
                raise ValueError(f"Could not find method {endpoint.name} on {impl}!!")

            impl_method = getattr(impl, endpoint.name)

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning, module="pydantic._internal._fields"
                )
                getattr(app, endpoint.method)(endpoint.route, response_model=None)(
                    create_dynamic_typed_route(
                        impl_method,
                        endpoint.method,
                    )
                )

        cprint(f"Serving API {api_str}", "white", attrs=["bold"])
        for endpoint in endpoints:
            cprint(f" {endpoint.method.upper()} {endpoint.route}", "white")

    print("")
    app.exception_handler(RequestValidationError)(global_exception_handler)
    app.exception_handler(Exception)(global_exception_handler)
    signal.signal(signal.SIGINT, functools.partial(handle_sigint, app))

    app.__llama_stack_impls__ = impls

    import uvicorn

    # FYI this does not do hot-reloads

    listen_host = ["::", "0.0.0.0"] if not args.disable_ipv6 else "0.0.0.0"
    print(f"Listening on {listen_host}:{args.port}")
    uvicorn.run(app, host=listen_host, port=args.port)


if __name__ == "__main__":
    main()
