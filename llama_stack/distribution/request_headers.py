# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import threading
from typing import Any, Dict

from .utils.dynamic import instantiate_class_type

_THREAD_LOCAL = threading.local()


class NeedsRequestProviderData:
    def get_request_provider_data(self) -> Any:
        spec = self.__provider_spec__
        assert spec, f"Provider spec not set on {self.__class__}"

        provider_type = spec.provider_type
        validator_class = spec.provider_data_validator
        if not validator_class:
            raise ValueError(f"Provider {provider_type} does not have a validator")

        val = getattr(_THREAD_LOCAL, "provider_data_header_value", None)
        if not val:
            return None

        validator = instantiate_class_type(validator_class)
        try:
            provider_data = validator(**val)
            return provider_data
        except Exception as e:
            print("Error parsing provider data", e)


def set_request_provider_data(headers: Dict[str, str]):
    keys = [
        "X-LlamaStack-ProviderData",
        "x-llamastack-providerdata",
    ]
    for key in keys:
        val = headers.get(key, None)
        if val:
            break

    if not val:
        return

    try:
        val = json.loads(val)
    except json.JSONDecodeError:
        print("Provider data not encoded as a JSON object!", val)
        return

    _THREAD_LOCAL.provider_data_header_value = val
