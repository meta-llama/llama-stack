# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum

from llama_models.schema_utils import json_schema_type

from pydantic import BaseModel


class LogFormat(Enum):
    TEXT = "text"
    JSON = "json"


@json_schema_type
class ConsoleConfig(BaseModel):
    log_format: LogFormat = LogFormat.TEXT
