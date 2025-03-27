# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any, Dict

from llama_stack.apis.datatypes import Api

from .config import PodmanAILabImplConfig


async def get_adapter_impl(config: PodmanAILabImplConfig, deps: Dict[Api, Any]):
    from .podman_ai_lab import PodmanAILabInferenceAdapter

    impl = PodmanAILabInferenceAdapter(config.url, deps[Api.models])
    await impl.initialize()
    return impl
