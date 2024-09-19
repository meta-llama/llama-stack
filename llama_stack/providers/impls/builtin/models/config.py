# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from llama_models.datatypes import ModelFamily

from llama_models.schema_utils import json_schema_type
from llama_models.sku_list import all_registered_models, resolve_model

from pydantic import BaseModel, Field, field_validator


@json_schema_type
class BuiltinImplConfig(BaseModel): ...
