# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, Optional

from llama_models.llama3_1.api.datatypes import URL

from pydantic import BaseModel

from strong_typing.schema import json_schema_type


@json_schema_type
class TrainEvalDatasetColumnType(Enum):
    dialog = "dialog"
    text = "text"
    media = "media"
    number = "number"
    json = "json"


@json_schema_type
class TrainEvalDataset(BaseModel):
    """Dataset to be used for training or evaluating language models."""

    # TODO(ashwin): figure out if we need to add an enum for a "dataset type"

    columns: Dict[str, TrainEvalDatasetColumnType]
    content_url: URL
    metadata: Optional[Dict[str, Any]] = None
