# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import inspect
import json
import signal
import traceback

from collections.abc import (
    AsyncGenerator as AsyncGeneratorABC,
    AsyncIterator as AsyncIteratorABC,
)
from contextlib import asynccontextmanager
from ssl import SSLError
from typing import Any, AsyncGenerator, AsyncIterator, Dict, get_type_hints, Optional

import fire
import httpx
import yaml

from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ValidationError
from termcolor import cprint
from typing_extensions import Annotated

from llama_stack.providers.utils.telemetry.tracing import (
    end_trace,
    setup_logger,
    SpanStatus,
    start_trace,
)
from llama_stack.distribution.datatypes import *  # noqa: F403

from llama_stack.distribution.request_headers import set_request_provider_data
from llama_stack.distribution.resolver import resolve_impls_with_routing

from .endpoints import get_all_api_endpoints


def is_async_iterator_type(typ):
    if hasattr(typ, "__origin__"):
        origin = typ.__origin__
        if isinstance(origin, type):
            return issubclass(
                origin,
                (AsyncIterator, AsyncGenerator, AsyncIteratorABC, AsyncGeneratorABC),
            )
        return False
    return isinstance(
        typ, (AsyncIterator, AsyncGenerator, AsyncIteratorABC, AsyncGeneratorABC)
    )


def create_sse_event(data: Any) -> str:
    if isinstance(data, BaseModel):
        data = data.json()
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


def handle_sigint(*args, **kwargs):
    print("SIGINT or CTRL-C detected. Exiting gracefully...")
    loop = asyncio.get_event_loop()
    for task in asyncio.all_tasks(loop):
        task.cancel()
    loop.stop()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up")
    yield
    print("Shutting down")


def create_dynamic_passthrough(
    downstream_url: str, downstream_headers: Optional[Dict[str, str]] = None
):
    async def endpoint(request: Request):
        return await passthrough(request, downstream_url, downstream_headers)

    return endpoint


def create_dynamic_typed_route(func: Any, method: str):
    hints = get_type_hints(func)
    response_model = hints.get("return")

    # NOTE: I think it is better to just add a method within each Api
    # "Protocol" / adapter-impl to tell what sort of a response this request
    # is going to produce. /chat_completion can produce a streaming or
    # non-streaming response depending on if request.stream is True / False.
    is_streaming = is_async_iterator_type(response_model)

    if is_streaming:

        async def endpoint(request: Request, **kwargs):
            await start_trace(func.__name__)

            set_request_provider_data(request.headers)

            async def sse_generator(event_gen):
                try:
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

            return StreamingResponse(
                sse_generator(func(**kwargs)), media_type="text/event-stream"
            )

    else:

        async def endpoint(request: Request, **kwargs):
            await start_trace(func.__name__)

            set_request_provider_data(request.headers)

            try:
                return (
                    await func(**kwargs)
                    if asyncio.iscoroutinefunction(func)
                    else func(**kwargs)
                )
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


def main(
    yaml_config: str = "llamastack-run.yaml",
    port: int = 5000,
    disable_ipv6: bool = False,
):
    with open(yaml_config, "r") as fp:
        config = StackRunConfig(**yaml.safe_load(fp))

    app = FastAPI()

    impls, specs = asyncio.run(resolve_impls_with_routing(config))
    if Api.telemetry in impls:
        setup_logger(impls[Api.telemetry])

    all_endpoints = get_all_api_endpoints()

    if config.apis_to_serve:
        apis_to_serve = set(config.apis_to_serve)
    else:
        apis_to_serve = set(impls.keys())

    apis_to_serve.add(Api.inspect)
    for api_str in apis_to_serve:
        api = Api(api_str)

        endpoints = all_endpoints[api]
        impl = impls[api]

        provider_spec = specs[api]
        if (
            isinstance(provider_spec, RemoteProviderSpec)
            and provider_spec.adapter is None
        ):
            for endpoint in endpoints:
                url = impl.__provider_config__.url.rstrip("/") + endpoint.route
                getattr(app, endpoint.method)(endpoint.route)(
                    create_dynamic_passthrough(url)
                )
        else:
            for endpoint in endpoints:
                if not hasattr(impl, endpoint.name):
                    # ideally this should be a typing violation already
                    raise ValueError(
                        f"Could not find method {endpoint.name} on {impl}!!"
                    )

                impl_method = getattr(impl, endpoint.name)

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
    signal.signal(signal.SIGINT, handle_sigint)

    import uvicorn

    # FYI this does not do hot-reloads
    listen_host = "::" if not disable_ipv6 else "0.0.0.0"
    print(f"Listening on {listen_host}:{port}")
    uvicorn.run(app, host=listen_host, port=port)


if __name__ == "__main__":
    fire.Fire(main)
