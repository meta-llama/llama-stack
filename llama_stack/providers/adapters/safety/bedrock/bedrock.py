# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.apis.safety import *  # noqa: F403
from llama_stack.providers.utils import get_request_provider_data

from .config import BedrockSafetyRequestProviderData


class BedrockSafetyAdapter(Safety):
    def __init__(self, url: str) -> None:
        self.url = url
        pass

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def run_shield(
        self,
        shield: ShieldType,
        messages: List[Message],
    ) -> RunShieldResponse:
        # clients will set api_keys by doing something like:
        #
        # client = llama_stack.LlamaStack()
        # await client.safety.run_shield(
        #   shield_type="aws_guardrail_type",
        #   messages=[ ... ],
        #   x_llamastack_provider_data={
        #     "aws_api_key": "..."
        #   }
        # )
        #
        # This information will arrive at the LlamaStack server via a HTTP Header.
        #
        # The server will then provide you a type-checked version of this provider data
        # automagically by extracting it from the header and validating it with the
        # BedrockSafetyRequestProviderData class you will need to register in the provider
        # registry.
        #
        provider_data: BedrockSafetyRequestProviderData = get_request_provider_data()
        # use `aws_api_key` to pass to the AWS servers in whichever form

        raise NotImplementedError()
