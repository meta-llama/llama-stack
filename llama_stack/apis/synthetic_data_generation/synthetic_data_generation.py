# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum

from typing import Any, Dict, List, Optional, Protocol

from llama_models.schema_utils import json_schema_type, webmethod

from pydantic import BaseModel

from llama_models.llama3.api.datatypes import *  # noqa: F403


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

    dialogs: List[Message]
    filtering_function: FilteringFunction = FilteringFunction.none
    model: Optional[str] = None


@json_schema_type
class SyntheticDataGenerationResponse(BaseModel):
    """Response from the synthetic data generation. Batch of (prompt, response, score) tuples that pass the threshold."""

    synthetic_data: List[Dict[str, Any]]
    statistics: Optional[Dict[str, Any]] = None


class SyntheticDataGeneration(Protocol):
    @webmethod(route="/synthetic_data_generation/generate")
    def synthetic_data_generate(
        self,
        dialogs: List[Message],
        filtering_function: FilteringFunction = FilteringFunction.none,
        model: Optional[str] = None,
    ) -> Union[SyntheticDataGenerationResponse]: ...
