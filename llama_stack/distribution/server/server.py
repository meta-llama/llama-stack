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
import ssl
import sys
import traceback
import warnings
from collections.abc import Callable
from contextlib import asynccontextmanager
from importlib.metadata import version as parse_version
from pathlib import Path
from typing import Annotated, Any, get_origin

import rich.pretty
import yaml
from aiohttp import hdrs
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi import Path as FastapiPath
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from openai import BadRequestError
from pydantic import BaseModel, ValidationError

from llama_stack.apis.common.responses import PaginatedResponse
from llama_stack.distribution.datatypes import AuthenticationRequiredError, LoggingConfig, StackRunConfig
from llama_stack.distribution.distribution import builtin_automatically_routed_apis
from llama_stack.distribution.request_headers import PROVIDER_DATA_VAR, User, request_provider_data_context
from llama_stack.distribution.resolver import InvalidProviderError
from llama_stack.distribution.server.routes import (
    find_matching_route,
    get_all_api_routes,
    initialize_route_impls,
)
from llama_stack.distribution.stack import (
    construct_stack,
    replace_env_vars,
    validate_env_pair,
)
from llama_stack.distribution.utils.config import redact_sensitive_fields
from llama_stack.distribution.utils.context import preserve_contexts_async_generator
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import Api
from llama_stack.providers.inline.telemetry.meta_reference.config import TelemetryConfig
from llama_stack.providers.inline.telemetry.meta_reference.telemetry import (
    TelemetryAdapter,
)
from llama_stack.providers.utils.telemetry.tracing import (
    CURRENT_TRACE_CONTEXT,
    end_trace,
    setup_logger,
    start_trace,
)

from .auth import AuthenticationMiddleware
from .quota import QuotaMiddleware

REPO_ROOT = Path(__file__).parent.parent.parent.parent

logger = get_logger(name=__name__, category="server")


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

    return JSONResponse(status_code=http_exc.status_code, content={"error": {"detail": http_exc.detail}})


def translate_exception(exc: Exception) -> HTTPException | RequestValidationError:
    if isinstance(exc, ValidationError):
        exc = RequestValidationError(exc.errors())

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
    elif isinstance(exc, BadRequestError):
        return HTTPException(status_code=400, detail=str(exc))
    elif isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail=f"Permission denied: {str(exc)}")
    elif isinstance(exc, asyncio.TimeoutError | TimeoutError):
        return HTTPException(status_code=504, detail=f"Operation timed out: {str(exc)}")
    elif isinstance(exc, NotImplementedError):
        return HTTPException(status_code=501, detail=f"Not implemented: {str(exc)}")
    elif isinstance(exc, AuthenticationRequiredError):
        return HTTPException(status_code=401, detail=f"Authentication required: {str(exc)}")
    else:
        return HTTPException(
            status_code=500,
            detail="Internal server error: An unexpected error occurred.",
        )


async def shutdown(app):
    """Initiate a graceful shutdown of the application.

    Handled by the lifespan context manager. The shutdown process involves
    shutting down all implementations registered in the application.
    """
    for impl in app.__llama_stack_impls__.values():
        impl_name = impl.__class__.__name__
        logger.info("Shutting down %s", impl_name)
        try:
            if hasattr(impl, "shutdown"):
                await asyncio.wait_for(impl.shutdown(), timeout=5)
            else:
                logger.warning("No shutdown method for %s", impl_name)
        except (asyncio.TimeoutError, TimeoutError):
            logger.exception("Shutdown timeout for %s ", impl_name, exc_info=True)
        except (Exception, asyncio.CancelledError) as e:
            logger.exception("Failed to shutdown %s: %s", impl_name, {e})


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up")
    yield
    logger.info("Shutting down")
    await shutdown(app)


def is_streaming_request(func_name: str, request: Request, **kwargs):
    # TODO: pass the api method and punt it to the Protocol definition directly
    return kwargs.get("stream", False)


async def maybe_await(value):
    if inspect.iscoroutine(value):
        return await value
    return value


async def sse_generator(event_gen_coroutine):
    event_gen = None
    try:
        event_gen = await event_gen_coroutine
        async for item in event_gen:
            yield create_sse_event(item)
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        logger.info("Generator cancelled")
        if event_gen:
            await event_gen.aclose()
    except Exception as e:
        logger.exception("Error in sse_generator")
        yield create_sse_event(
            {
                "error": {
                    "message": str(translate_exception(e)),
                },
            }
        )


async def log_request_pre_validation(request: Request):
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            body_bytes = await request.body()
            if body_bytes:
                try:
                    parsed_body = json.loads(body_bytes.decode())
                    log_output = rich.pretty.pretty_repr(parsed_body)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    log_output = repr(body_bytes)
                logger.debug(f"Incoming raw request body for {request.method} {request.url.path}:\n{log_output}")
            else:
                logger.debug(f"Incoming {request.method} {request.url.path} request with empty body.")
        except Exception as e:
            logger.warning(f"Could not read or log request body for {request.method} {request.url.path}: {e}")


def create_dynamic_typed_route(func: Any, method: str, route: str) -> Callable:
    @functools.wraps(func)
    async def route_handler(request: Request, **kwargs):
        # Get auth attributes from the request scope
        user_attributes = request.scope.get("user_attributes", {})
        principal = request.scope.get("principal", "")
        user = User(principal, user_attributes)

        await log_request_pre_validation(request)

        # Use context manager with both provider data and auth attributes
        with request_provider_data_context(request.headers, user):
            is_streaming = is_streaming_request(func.__name__, request, **kwargs)

            try:
                if is_streaming:
                    gen = preserve_contexts_async_generator(
                        sse_generator(func(**kwargs)), [CURRENT_TRACE_CONTEXT, PROVIDER_DATA_VAR]
                    )
                    return StreamingResponse(gen, media_type="text/event-stream")
                else:
                    value = func(**kwargs)
                    result = await maybe_await(value)
                    if isinstance(result, PaginatedResponse) and result.url is None:
                        result.url = route
                    return result
            except Exception as e:
                logger.exception(f"Error executing endpoint {route=} {method=}")
                raise translate_exception(e) from e

    sig = inspect.signature(func)

    new_params = [inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Request)]
    new_params.extend(sig.parameters.values())

    path_params = extract_path_params(route)
    if method == "post":
        # Annotate parameters that are in the path with Path(...) and others with Body(...),
        # but preserve existing File() and Form() annotations for multipart form data
        new_params = (
            [new_params[0]]
            + [
                (
                    param.replace(annotation=Annotated[param.annotation, FastapiPath(..., title=param.name)])
                    if param.name in path_params
                    else (
                        param  # Keep original annotation if it's already an Annotated type
                        if get_origin(param.annotation) is Annotated
                        else param.replace(annotation=Annotated[param.annotation, Body(..., embed=True)])
                    )
                )
                for param in new_params[1:]
            ]
        )

    route_handler.__signature__ = sig.replace(parameters=new_params)

    return route_handler


class TracingMiddleware:
    def __init__(self, app, impls):
        self.app = app
        self.impls = impls
        # FastAPI built-in paths that should bypass custom routing
        self.fastapi_paths = ("/docs", "/redoc", "/openapi.json", "/favicon.ico", "/static")

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "lifespan":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")

        # Check if the path is a FastAPI built-in path
        if path.startswith(self.fastapi_paths):
            # Pass through to FastAPI's built-in handlers
            logger.debug(f"Bypassing custom routing for FastAPI built-in path: {path}")
            return await self.app(scope, receive, send)

        if not hasattr(self, "route_impls"):
            self.route_impls = initialize_route_impls(self.impls)

        try:
            _, _, trace_path = find_matching_route(scope.get("method", hdrs.METH_GET), path, self.route_impls)
        except ValueError:
            # If no matching endpoint is found, pass through to FastAPI
            logger.debug(f"No matching route found for path: {path}, falling back to FastAPI")
            return await self.app(scope, receive, send)

        trace_attributes = {"__location__": "server", "raw_path": path}

        # Extract W3C trace context headers and store as trace attributes
        headers = dict(scope.get("headers", []))
        traceparent = headers.get(b"traceparent", b"").decode()
        if traceparent:
            trace_attributes["traceparent"] = traceparent
        tracestate = headers.get(b"tracestate", b"").decode()
        if tracestate:
            trace_attributes["tracestate"] = tracestate

        trace_context = await start_trace(trace_path, trace_attributes)

        async def send_with_trace_id(message):
            if message["type"] == "http.response.start":
                headers = message.get("headers", [])
                headers.append([b"x-trace-id", str(trace_context.trace_id).encode()])
                message["headers"] = headers
            await send(message)

        try:
            return await self.app(scope, receive, send_with_trace_id)
        finally:
            await end_trace()


class ClientVersionMiddleware:
    def __init__(self, app):
        self.app = app
        self.server_version = parse_version("llama-stack")

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            client_version = headers.get(b"x-llamastack-client-version", b"").decode()
            if client_version:
                try:
                    client_version_parts = tuple(map(int, client_version.split(".")[:2]))
                    server_version_parts = tuple(map(int, self.server_version.split(".")[:2]))
                    if client_version_parts != server_version_parts:

                        async def send_version_error(send):
                            await send(
                                {
                                    "type": "http.response.start",
                                    "status": 426,
                                    "headers": [[b"content-type", b"application/json"]],
                                }
                            )
                            error_msg = json.dumps(
                                {
                                    "error": {
                                        "message": f"Client version {client_version} is not compatible with server version {self.server_version}. Please update your client."
                                    }
                                }
                            ).encode()
                            await send({"type": "http.response.body", "body": error_msg})

                        return await send_version_error(send)
                except (ValueError, IndexError):
                    # If version parsing fails, let the request through
                    pass

        return await self.app(scope, receive, send)


def main(args: argparse.Namespace | None = None):
    """Start the LlamaStack server."""
    parser = argparse.ArgumentParser(description="Start the LlamaStack server.")
    parser.add_argument(
        "--yaml-config",
        dest="config",
        help="(Deprecated) Path to YAML configuration file - use --config instead",
    )
    parser.add_argument(
        "--config",
        dest="config",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--template",
        help="One of the template names in llama_stack/templates (e.g., tgi, fireworks, remote-vllm, etc.)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("LLAMA_STACK_PORT", 8321)),
        help="Port to listen on",
    )
    parser.add_argument(
        "--env",
        action="append",
        help="Environment variables in KEY=value format. Can be specified multiple times.",
    )

    # Determine whether the server args are being passed by the "run" command, if this is the case
    # the args will be passed as a Namespace object to the main function, otherwise they will be
    # parsed from the command line
    if args is None:
        args = parser.parse_args()

    log_line = ""
    if args.config:
        # if the user provided a config file, use it, even if template was specified
        config_file = Path(args.config)
        if not config_file.exists():
            raise ValueError(f"Config file {config_file} does not exist")
        log_line = f"Using config file: {config_file}"
    elif args.template:
        config_file = Path(REPO_ROOT) / "llama_stack" / "templates" / args.template / "run.yaml"
        if not config_file.exists():
            raise ValueError(f"Template {args.template} does not exist")
        log_line = f"Using template {args.template} config file: {config_file}"
    else:
        raise ValueError("Either --config or --template must be provided")

    logger_config = None
    with open(config_file) as fp:
        config_contents = yaml.safe_load(fp)
        if isinstance(config_contents, dict) and (cfg := config_contents.get("logging_config")):
            logger_config = LoggingConfig(**cfg)
        logger = get_logger(name=__name__, category="server", config=logger_config)
        if args.env:
            for env_pair in args.env:
                try:
                    key, value = validate_env_pair(env_pair)
                    logger.info(f"Setting CLI environment variable {key} => {value}")
                    os.environ[key] = value
                except ValueError as e:
                    logger.error(f"Error: {str(e)}")
                    sys.exit(1)
        config = replace_env_vars(config_contents)
        config = StackRunConfig(**config)

    # now that the logger is initialized, print the line about which type of config we are using.
    logger.info(log_line)

    logger.info("Run configuration:")
    safe_config = redact_sensitive_fields(config.model_dump())
    logger.info(yaml.dump(safe_config, indent=2))

    app = FastAPI(
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    if not os.environ.get("LLAMA_STACK_DISABLE_VERSION_CHECK"):
        app.add_middleware(ClientVersionMiddleware)

    # Add authentication middleware if configured
    if config.server.auth:
        logger.info(f"Enabling authentication with provider: {config.server.auth.provider_type.value}")
        app.add_middleware(AuthenticationMiddleware, auth_config=config.server.auth)
    else:
        if config.server.quota:
            quota = config.server.quota
            logger.warning(
                "Configured authenticated_max_requests (%d) but no auth is enabled; "
                "falling back to anonymous_max_requests (%d) for all the requests",
                quota.authenticated_max_requests,
                quota.anonymous_max_requests,
            )

    if config.server.quota:
        logger.info("Enabling quota middleware for authenticated and anonymous clients")

        quota = config.server.quota
        anonymous_max_requests = quota.anonymous_max_requests
        # if auth is disabled, use the anonymous max requests
        authenticated_max_requests = quota.authenticated_max_requests if config.server.auth else anonymous_max_requests

        kv_config = quota.kvstore
        window_map = {"day": 86400}
        window_seconds = window_map[quota.period.value]

        app.add_middleware(
            QuotaMiddleware,
            kv_config=kv_config,
            anonymous_max_requests=anonymous_max_requests,
            authenticated_max_requests=authenticated_max_requests,
            window_seconds=window_seconds,
        )

    try:
        impls = asyncio.run(construct_stack(config))
    except InvalidProviderError as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

    if Api.telemetry in impls:
        setup_logger(impls[Api.telemetry])
    else:
        setup_logger(TelemetryAdapter(TelemetryConfig(), {}))

    all_routes = get_all_api_routes()

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
    apis_to_serve.add("providers")
    for api_str in apis_to_serve:
        api = Api(api_str)

        routes = all_routes[api]
        impl = impls[api]

        for route in routes:
            if not hasattr(impl, route.name):
                # ideally this should be a typing violation already
                raise ValueError(f"Could not find method {route.name} on {impl}!")

            impl_method = getattr(impl, route.name)
            # Filter out HEAD method since it's automatically handled by FastAPI for GET routes
            available_methods = [m for m in route.methods if m != "HEAD"]
            if not available_methods:
                raise ValueError(f"No methods found for {route.name} on {impl}")
            method = available_methods[0]
            logger.debug(f"{method} {route.path}")

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._fields")
                getattr(app, method.lower())(route.path, response_model=None)(
                    create_dynamic_typed_route(
                        impl_method,
                        method.lower(),
                        route.path,
                    )
                )

    logger.debug(f"serving APIs: {apis_to_serve}")

    app.exception_handler(RequestValidationError)(global_exception_handler)
    app.exception_handler(Exception)(global_exception_handler)

    app.__llama_stack_impls__ = impls
    app.add_middleware(TracingMiddleware, impls=impls)

    import uvicorn

    # Configure SSL if certificates are provided
    port = args.port or config.server.port

    ssl_config = None
    keyfile = config.server.tls_keyfile
    certfile = config.server.tls_certfile

    if keyfile and certfile:
        ssl_config = {
            "ssl_keyfile": keyfile,
            "ssl_certfile": certfile,
        }
        if config.server.tls_cafile:
            ssl_config["ssl_ca_certs"] = config.server.tls_cafile
            ssl_config["ssl_cert_reqs"] = ssl.CERT_REQUIRED
            logger.info(
                f"HTTPS enabled with certificates:\n  Key: {keyfile}\n  Cert: {certfile}\n  CA: {config.server.tls_cafile}"
            )
        else:
            logger.info(f"HTTPS enabled with certificates:\n  Key: {keyfile}\n  Cert: {certfile}")

    listen_host = config.server.host or ["::", "0.0.0.0"]
    logger.info(f"Listening on {listen_host}:{port}")

    uvicorn_config = {
        "app": app,
        "host": listen_host,
        "port": port,
        "lifespan": "on",
        "log_level": logger.getEffectiveLevel(),
    }
    if ssl_config:
        uvicorn_config.update(ssl_config)

    uvicorn.run(**uvicorn_config)


def extract_path_params(route: str) -> list[str]:
    segments = route.split("/")
    params = [seg[1:-1] for seg in segments if seg.startswith("{") and seg.endswith("}")]
    # to handle path params like {param:path}
    params = [param.split(":")[0] for param in params]
    return params


if __name__ == "__main__":
    main()
