# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .datatypes import *  # noqa: F403
from typing import List, Protocol

from llama_models.llama3_1.api.datatypes import Message

# this dependency is annoying and we need a forked up version anyway
from llama_models.schema_utils import webmethod


@json_schema_type
class RunShieldRequest(BaseModel):
    messages: List[Message]
    shields: List[ShieldDefinition]


@json_schema_type
class RunShieldResponse(BaseModel):
    responses: List[ShieldResponse]


class Safety(Protocol):

    @webmethod(route="/safety/run_shields")
    async def run_shields(
        self,
        request: RunShieldRequest,
    ) -> RunShieldResponse: ...
