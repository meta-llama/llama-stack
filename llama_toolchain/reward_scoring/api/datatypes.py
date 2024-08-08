# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_models.schema_utils import json_schema_type

from pydantic import BaseModel

from llama_models.llama3_1.api.datatypes import *  # noqa: F403


@json_schema_type
class ScoredMessage(BaseModel):
    message: Message
    score: float


@json_schema_type
class DialogGenerations(BaseModel):
    dialog: List[Message]
    sampled_generations: List[Message]


@json_schema_type
class ScoredDialogGenerations(BaseModel):
    dialog: List[Message]
    scored_generations: List[ScoredMessage]
