# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, List, Optional

from llama_models.schema_utils import json_schema_type
from llama_models.sku_list import all_registered_models, resolve_model
from llama_stack.distribution.datatypes import GenericProviderConfig

from pydantic import BaseModel, Field, field_validator


@json_schema_type
class ModelConfigProviderEntry(GenericProviderConfig):
    api: str
    core_model_id: str


@json_schema_type
class BuiltinImplConfig(BaseModel):
    models_config: List[ModelConfigProviderEntry]
