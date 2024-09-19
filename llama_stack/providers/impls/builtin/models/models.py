# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio

from typing import AsyncIterator, Union

from llama_models.llama3.api.datatypes import StopReason
from llama_models.sku_list import resolve_model

from llama_stack.apis.models import *  # noqa: F403
from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_models.datatypes import CoreModelId, Model
from llama_models.sku_list import resolve_model

from .config import BuiltinImplConfig

DUMMY_MODELS_SPEC = ModelSpec(
    llama_model_metadata=resolve_model("Meta-Llama3.1-8B"),
    providers_spec={"inference": {"provider_type": "meta-reference"}},
)


class BuiltinModelsImpl(Models):
    def __init__(
        self,
        config: BuiltinImplConfig,
    ) -> None:
        self.config = config
        self.models_list = [DUMMY_MODELS_SPEC]

    async def initialize(self) -> None:
        pass

    async def list_models(self) -> ModelsListResponse:
        return ModelsListResponse(models_list=self.models_list)

    async def get_model(self, core_model_id: str) -> ModelsGetResponse:
        return ModelsGetResponse(core_model_spec=DUMMY_MODELS_SPEC)

    async def register_model(
        self, model_id: str, api: str, provider_spec: Dict[str, str]
    ) -> ModelsRegisterResponse:
        return ModelsRegisterResponse()
