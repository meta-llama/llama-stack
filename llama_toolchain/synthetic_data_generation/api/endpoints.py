# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel

from pyopenapi import webmethod
from strong_typing.schema import json_schema_type

from llama_models.llama3_1.api.datatypes import *  # noqa: F403
from llama_toolchain.reward_scoring.api.datatypes import *  # noqa: F403
from .datatypes import *  # noqa: F403


@json_schema_type
class SyntheticDataGenerationRequest(BaseModel):
    """Request to generate synthetic data. A small batch of prompts and a filtering function"""

    dialogs: List[Message]
    filtering_function: FilteringFunction = FilteringFunction.none
    model: Optional[RewardModel] = None


@json_schema_type
class SyntheticDataGenerationResponse(BaseModel):
    """Response from the synthetic data generation. Batch of (prompt, response, score) tuples that pass the threshold."""

    synthetic_data: List[ScoredDialogGenerations]
    statistics: Optional[Dict[str, Any]] = None


class SyntheticDataGeneration(Protocol):
    @webmethod(route="/synthetic_data_generation/generate")
    def post_generate(
        self,
        request: SyntheticDataGenerationRequest,
    ) -> Union[SyntheticDataGenerationResponse]: ...
