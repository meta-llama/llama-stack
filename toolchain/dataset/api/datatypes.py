from enum import Enum
from typing import Any, Dict, Optional

from models.llama3_1.api.datatypes import URL

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
