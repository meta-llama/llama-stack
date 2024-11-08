# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.eval import EvalTaskDef

meta_reference_mmlu = EvalTaskDef(
    identifier="meta-reference-mmlu",
    dataset_id="llamastack_mmlu_loose",
    scoring_functions=["meta-reference::regex_parser_multiple_choice_answer"],
)
