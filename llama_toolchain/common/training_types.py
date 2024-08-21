# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_models.llama3.api.datatypes import URL
from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel


@json_schema_type(schema={"description": "Checkpoint created during training runs"})
class Checkpoint(BaseModel):
    iters: int
    path: URL
    epoch: int
