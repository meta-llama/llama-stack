# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.common.type_system import NumberType
from llama_stack.apis.scoring_functions import ScoringFnDef


equality = ScoringFnDef(
    identifier="meta-reference::equality",
    description="Returns 1.0 if the input is equal to the target, 0.0 otherwise.",
    parameters=[],
    return_type=NumberType(),
)
