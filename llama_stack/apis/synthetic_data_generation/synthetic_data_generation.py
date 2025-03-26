# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel

from llama_stack.apis.inference import Message
from llama_stack.schema_utils import json_schema_type, webmethod


class FilteringFunction(Enum):
    """The type of filtering function."""

    none = "none"
    random = "random"
    top_k = "top_k"
    top_p = "top_p"
    top_k_top_p = "top_k_top_p"
    sigmoid = "sigmoid"


@json_schema_type
class SyntheticDataGenerationRequest(BaseModel):
    """Request to generate synthetic data. A small batch of prompts and a filtering function"""

    dialogs: list[Message]
    filtering_function: FilteringFunction = FilteringFunction.none
    model: str | None = None


@json_schema_type
class SyntheticDataGenerationResponse(BaseModel):
    """Response from the synthetic data generation. Batch of (prompt, response, score) tuples that pass the threshold."""

    synthetic_data: list[dict[str, Any]]
    statistics: dict[str, Any] | None = None


class SyntheticDataGeneration(Protocol):
    @webmethod(route="/synthetic-data-generation/generate")
    def synthetic_data_generate(
        self,
        dialogs: list[Message],
        filtering_function: FilteringFunction = FilteringFunction.none,
        model: str | None = None,
    ) -> SyntheticDataGenerationResponse: ...
