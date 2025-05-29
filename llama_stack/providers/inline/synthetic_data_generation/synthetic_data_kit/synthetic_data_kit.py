# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from typing import Optional
import synthetic_data_kit as sdk

from llama_stack.apis.inference import Message
from llama_stack.apis.synthetic_data_generation import (
    FilteringFunction,
    SyntheticDataGeneration,
    SyntheticDataGenerationResponse,
)

from .config import SyntheticDataKitConfig


class SyntheticDataKitProvider(SyntheticDataGeneration):
    def __init__(self, config: SyntheticDataKitConfig):
        self.config = config
        self.sdk = sdk.SyntheticDataKit(
            llm=self.config.llm,
            vllm=self.config.vllm,
            generation=self.config.generation,
            curate=self.config.curate,
        )

    async def synthetic_data_generate(
        self,
        dialogs: list[Message],
        filtering_function: FilteringFunction = FilteringFunction.none,
        model: Optional[str] = None,
    ) -> SyntheticDataGenerationResponse:
        # Convert dialogs to text format
        text_content = "\n".join(d.content for d in dialogs)
        
        # Generate synthetic data
        if filtering_function == FilteringFunction.none:
            result = await self.sdk.create(text_content, type="qa")
        else:
            # Generate and then curate
            generated = await self.sdk.create(text_content, type="qa")
            result = await self.sdk.curate(generated)

        return SyntheticDataGenerationResponse(
            synthetic_data=result.get("synthetic_data", []),
            statistics=result.get("statistics"),
        )
