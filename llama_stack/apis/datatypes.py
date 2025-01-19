# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum

from llama_models.schema_utils import json_schema_type


@json_schema_type
class Api(Enum):
    inference = "inference"
    safety = "safety"
    agents = "agents"
    vector_io = "vector_io"
    datasetio = "datasetio"
    scoring = "scoring"
    eval = "eval"
    post_training = "post_training"
    tool_runtime = "tool_runtime"

    telemetry = "telemetry"

    models = "models"
    shields = "shields"
    vector_dbs = "vector_dbs"
    datasets = "datasets"
    scoring_functions = "scoring_functions"
    eval_tasks = "eval_tasks"
    tool_groups = "tool_groups"

    # built-in API
    inspect = "inspect"
