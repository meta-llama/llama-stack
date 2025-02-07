# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel

from llama_models.schema_utils import json_schema_type


@json_schema_type
class FiddlecubeSafetyConfig(BaseModel):
    pass
