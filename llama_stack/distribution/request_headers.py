# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import contextvars
import json
import logging
from typing import Any, AsyncGenerator, ContextManager, Dict, Optional, TypeVar

from .utils.dynamic import instantiate_class_type

log = logging.getLogger(__name__)

# Context variable for request provider data
_provider_data_var = contextvars.ContextVar("provider_data", default=None)


class RequestProviderDataContext(ContextManager):
    """Context manager for request provider data"""

    def __init__(self, provider_data: Optional[Dict[str, Any]] = None):
        self.provider_data = provider_data
        self.token = None

    def __enter__(self):
        # Save the current value and set the new one
        self.token = _provider_data_var.set(self.provider_data)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the previous value
        if self.token is not None:
            _provider_data_var.reset(self.token)


T = TypeVar("T")


def preserve_headers_context_async_generator(gen: AsyncGenerator[T, None]) -> AsyncGenerator[T, None]:
    """
    Wraps an async generator to preserve request headers context variables across iterations.

    This ensures that context variables set during generator creation are
    available during each iteration of the generator, even if the original
    context manager has exited.
    """
    # Capture the current context value right now
    context_value = _provider_data_var.get()

    async def wrapper():
        while True:
            # Set context before each anext() call
            token = _provider_data_var.set(context_value)
            try:
                item = await gen.__anext__()
                yield item
            except StopAsyncIteration:
                break
            finally:
                # Restore the previous value
                _provider_data_var.reset(token)

    return wrapper()


class NeedsRequestProviderData:
    def get_request_provider_data(self) -> Any:
        spec = self.__provider_spec__
        assert spec, f"Provider spec not set on {self.__class__}"

        provider_type = spec.provider_type
        validator_class = spec.provider_data_validator
        if not validator_class:
            raise ValueError(f"Provider {provider_type} does not have a validator")

        val = _provider_data_var.get()
        if not val:
            return None

        validator = instantiate_class_type(validator_class)
        try:
            provider_data = validator(**val)
            return provider_data
        except Exception as e:
            log.error(f"Error parsing provider data: {e}")
            return None


def parse_request_provider_data(headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Parse provider data from request headers"""
    keys = [
        "X-LlamaStack-Provider-Data",
        "x-llamastack-provider-data",
    ]
    val = None
    for key in keys:
        val = headers.get(key, None)
        if val:
            break

    if not val:
        return None

    try:
        return json.loads(val)
    except json.JSONDecodeError:
        log.error("Provider data not encoded as a JSON object!")
        return None


def request_provider_data_context(headers: Dict[str, str]) -> ContextManager:
    """Context manager that sets request provider data from headers for the duration of the context"""
    provider_data = parse_request_provider_data(headers)
    return RequestProviderDataContext(provider_data)
