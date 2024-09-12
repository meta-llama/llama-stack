# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol

from llama_models.schema_utils import webmethod  # noqa: F401

from pydantic import BaseModel  # noqa: F401


@json_schema_type
class ModelSpec(BaseModel):
    model_name: str = Field(description="Name of the model")
    providers_spec: Dict[Api, Dict[str, str]] = Field(
        default_factory=dict,
        description="Map of API to the concrete provider specs. E.g. {}".format(
            {
                "inference": {
                    "provider_type": "remote::tgi",
                    "url": "localhost::5555",
                    "api_token": "hf_xxx",
                }
            }
        ),
    )


class Models(Protocol):
    @webmethod(route="/models/list", method="GET")
    async def list_models(self) -> List[ModelSpec]: ...

    @webmethod(route="/models/get", method="GET")
    async def get_model(self, model_name: str) -> ModelSpec: ...

    @webmethod(route="/models/register")
    async def register_model(
        self, name: str, provider: Api, provider_spec: Dict[str, str]
    ) -> ModelSpec: ...
