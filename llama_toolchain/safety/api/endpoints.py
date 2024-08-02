# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .datatypes import *  # noqa: F403
from typing import List, Protocol

from llama_models.llama3_1.api.datatypes import Message

# this dependency is annoying and we need a forked up version anyway
from pyopenapi import webmethod


@json_schema_type
class RunShieldRequest(BaseModel):
    shield_type: ShieldType
    messages: List[Message]


class Safety(Protocol):

    @webmethod(route="/safety/run_shield")
    async def run_shield(
        self,
        request: RunShieldRequest,
    ) -> ShieldResponse: ...
