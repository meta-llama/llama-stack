# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.scoring_functions import *  # noqa: F401, F403
from llama_stack.apis.scoring import *  # noqa: F401, F403
from llama_stack.apis.common.type_system import NumberType


llm_as_judge_base = ScoringFnDef(
    identifier="meta-reference::llm_as_judge_base",
    description="Llm As Judge Scoring Function",
    return_type=NumberType(),
)
