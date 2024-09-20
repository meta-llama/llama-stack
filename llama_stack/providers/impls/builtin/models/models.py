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
from termcolor import cprint

from .config import BuiltinImplConfig


class BuiltinModelsImpl(Models):
    def __init__(
        self,
        config: BuiltinImplConfig,
    ) -> None:
        self.config = config
        cprint(self.config, "red")
        self.models = {
            entry.core_model_id: ModelSpec(
                llama_model_metadata=resolve_model(entry.core_model_id),
                provider_id=entry.provider_id,
                api=entry.api,
                provider_config=entry.config,
            )
            for entry in self.config.models_config
        }

    async def initialize(self) -> None:
        pass

    async def list_models(self) -> ModelsListResponse:
        return ModelsListResponse(models_list=list(self.models.values()))

    async def get_model(self, core_model_id: str) -> ModelsGetResponse:
        if core_model_id in self.models:
            return ModelsGetResponse(core_model_spec=self.models[core_model_id])
        raise RuntimeError(f"Cannot find {core_model_id} in model registry")

    async def register_model(
        self, model_id: str, api: str, provider_spec: Dict[str, str]
    ) -> ModelsRegisterResponse:
        return ModelsRegisterResponse()
