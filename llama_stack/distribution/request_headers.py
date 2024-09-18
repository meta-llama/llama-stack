# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import threading
from typing import Any, Dict, Optional

from .utils.dynamic import instantiate_class_type

_THREAD_LOCAL = threading.local()


def get_request_provider_data() -> Any:
    return getattr(_THREAD_LOCAL, "provider_data", None)


def set_request_provider_data(headers: Dict[str, str], validator_class: Optional[str]):
    if not validator_class:
        return

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

    validator = instantiate_class_type(validator_class)
    try:
        provider_data = validator(**val)
    except Exception as e:
        print("Error parsing provider data", e)
        return

    _THREAD_LOCAL.provider_data = provider_data
