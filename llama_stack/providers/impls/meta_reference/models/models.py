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
from llama_stack.apis.inference import Inference
from llama_stack.apis.safety import Safety

from llama_stack.providers.impls.meta_reference.inference.inference import (
    MetaReferenceInferenceImpl,
)
from llama_stack.providers.impls.meta_reference.safety.safety import (
    MetaReferenceSafetyImpl,
)

from .config import MetaReferenceImplConfig


class MetaReferenceModelsImpl(Models):
    def __init__(
        self,
        config: MetaReferenceImplConfig,
        inference_api: Inference,
        safety_api: Safety,
    ) -> None:
        self.config = config
        self.inference_api = inference_api
        self.safety_api = safety_api

        self.models_list = []
        # TODO, make the inference route provider and use router provider to do the lookup dynamically
        if isinstance(
            self.inference_api,
            MetaReferenceInferenceImpl,
        ):
            model = resolve_model(self.inference_api.config.model)
            self.models_list.append(
                ModelSpec(
                    llama_model_metadata=model,
                    providers_spec={
                        "inference": [{"provider_type": "meta-reference"}],
                    },
                )
            )

        if isinstance(
            self.safety_api,
            MetaReferenceSafetyImpl,
        ):
            shield_cfg = self.safety_api.config.llama_guard_shield
            if shield_cfg is not None:
                model = resolve_model(shield_cfg.model)
                self.models_list.append(
                    ModelSpec(
                        llama_model_metadata=model,
                        providers_spec={
                            "safety": [{"provider_type": "meta-reference"}],
                        },
                    )
                )
            shield_cfg = self.safety_api.config.prompt_guard_shield
            if shield_cfg is not None:
                model = resolve_model(shield_cfg.model)
                self.models_list.append(
                    ModelSpec(
                        llama_model_metadata=model,
                        providers_spec={
                            "safety": [{"provider_type": "meta-reference"}],
                        },
                    )
                )

    async def initialize(self) -> None:
        pass

    async def list_models(self) -> ModelsListResponse:
        return ModelsListResponse(models_list=self.models_list)

    async def get_model(self, model_id: str) -> ModelsGetResponse:
        for model in self.models_list:
            if model.llama_model_metadata.core_model_id.value == model_id:
                return ModelsGetResponse(core_model_spec=model)
        return ModelsGetResponse()

    async def register_model(
        self, model_id: str, api: str, provider_spec: Dict[str, str]
    ) -> ModelsRegisterResponse:
        return ModelsRegisterResponse()
