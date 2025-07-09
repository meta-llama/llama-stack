# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime

from pydantic import BaseModel

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class PostTrainingMetric(BaseModel):
    epoch: int
    train_loss: float
    validation_loss: float
    perplexity: float


@json_schema_type
class Checkpoint(BaseModel):
    """Checkpoint created during training runs"""

    identifier: str
    created_at: datetime
    epoch: int
    post_training_job_id: str
    path: str
    training_metrics: PostTrainingMetric | None = None
