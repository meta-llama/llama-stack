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
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    get_type_hints,
    List,
    Optional,
    Set,
)

import fire
import httpx
import yaml

from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute
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

from llama_stack.distribution.distribution import api_endpoints, api_providers
from llama_stack.distribution.utils.dynamic import instantiate_provider


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


def translate_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, ValidationError):
        return RequestValidationError(exc.raw_errors)

    # Add more custom exception translations here
    return HTTPException(status_code=500, detail="Internal server error")


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

        async def endpoint(**kwargs):
            await start_trace(func.__name__)

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

        async def endpoint(**kwargs):
            await start_trace(func.__name__)
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
    if method == "post":
        # make sure every parameter is annotated with Body() so FASTAPI doesn't
        # do anything too intelligent and ask for some parameters in the query
        # and some in the body
        endpoint.__signature__ = sig.replace(
            parameters=[
                param.replace(
                    annotation=Annotated[param.annotation, Body(..., embed=True)]
                )
                for param in sig.parameters.values()
            ]
        )
    else:
        endpoint.__signature__ = sig

    return endpoint


def topological_sort(providers: List[ProviderSpec]) -> List[ProviderSpec]:
    by_id = {x.api: x for x in providers}

    def dfs(a: ProviderSpec, visited: Set[Api], stack: List[Api]):
        visited.add(a.api)

        for api in a.api_dependencies:
            if api not in visited:
                dfs(by_id[api], visited, stack)

        stack.append(a.api)

    visited = set()
    stack = []

    for a in providers:
        if a.api not in visited:
            dfs(a, visited, stack)

    return [by_id[x] for x in stack]


def snake_to_camel(snake_str):
    return "".join(word.capitalize() for word in snake_str.split("_"))


async def resolve_impls(
    provider_map: Dict[str, ProviderMapEntry],
) -> Dict[Api, Any]:
    """
    Does two things:
    - flatmaps, sorts and resolves the providers in dependency order
    - for each API, produces either a (local, passthrough or router) implementation
    """
    all_providers = api_providers()

    specs = {}
    for api_str, item in provider_map.items():
        api = Api(api_str)
        providers = all_providers[api]

        if isinstance(item, GenericProviderConfig):
            if item.provider_id not in providers:
                raise ValueError(
                    f"Unknown provider `{provider_id}` is not available for API `{api}`"
                )
            specs[api] = providers[item.provider_id]
        else:
            assert isinstance(item, list)
            inner_specs = []
            for rt_entry in item:
                if rt_entry.provider_id not in providers:
                    raise ValueError(
                        f"Unknown provider `{rt_entry.provider_id}` is not available for API `{api}`"
                    )
                inner_specs.append(providers[rt_entry.provider_id])

            specs[api] = RouterProviderSpec(
                api=api,
                module=f"llama_stack.providers.routers.{api.value.lower()}",
                api_dependencies=[],
                inner_specs=inner_specs,
            )

    sorted_specs = topological_sort(specs.values())

    impls = {}
    for spec in sorted_specs:
        api = spec.api

        deps = {api: impls[api] for api in spec.api_dependencies}
        impl = await instantiate_provider(spec, deps, provider_map[api.value])
        impls[api] = impl

    return impls, specs


def main(yaml_config: str, port: int = 5000, disable_ipv6: bool = False):
    with open(yaml_config, "r") as fp:
        config = StackRunConfig(**yaml.safe_load(fp))

    app = FastAPI()

    impls, specs = asyncio.run(resolve_impls(config.provider_map))
    if Api.telemetry in impls:
        setup_logger(impls[Api.telemetry])

    all_endpoints = api_endpoints()

    apis_to_serve = config.apis_to_serve or list(config.provider_map.keys())
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
                    create_dynamic_typed_route(impl_method, endpoint.method)
                )

    for route in app.routes:
        if isinstance(route, APIRoute):
            cprint(
                f"Serving {next(iter(route.methods))} {route.path}",
                "white",
                attrs=["bold"],
            )

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
