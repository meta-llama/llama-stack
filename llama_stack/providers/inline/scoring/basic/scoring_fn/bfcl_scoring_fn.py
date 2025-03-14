# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import re
from typing import Any, Dict, Optional

from llama_stack.apis.scoring import ScoringResultRow
from llama_stack.apis.scoring_functions import ScoringFnParams
from llama_stack.providers.utils.scoring.base_scoring_fn import RegisteredBaseScoringFn

from ..utils.bfcl.ast_parser import decode_ast
from ..utils.bfcl.checker import ast_checker, is_empty_output
from .fn_defs.bfcl import bfcl


def postprocess(x: Dict[str, Any], test_category: str) -> Dict[str, Any]:
    contain_func_call = False
    error = None
    error_type = None
    checker_result = {}
    try:
        prediction = decode_ast(x["generated_answer"], x["language"]) or ""
        contain_func_call = True
        # if not is_function_calling_format_output(prediction):
        if is_empty_output(prediction):
            contain_func_call = False
            error = "Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."
            error_type = "ast_decoder:decoder_wrong_output_format"
        else:
            checker_result = ast_checker(
                json.loads(x["function"]),
                prediction,
                json.loads(x["ground_truth"]),
                x["language"],
                test_category=test_category,
                model_name="",
            )
    except Exception as e:
        prediction = ""
        error = f"Invalid syntax. Failed to decode AST. {str(e)}"
        error_type = "ast_decoder:decoder_failed"
    return {
        "prediction": prediction,
        "contain_func_call": contain_func_call,
        "valid": checker_result.get("valid", False),
        "error": error or checker_result.get("error", ""),
        "error_type": error_type or checker_result.get("error_type", ""),
    }


def gen_valid(x: Dict[str, Any]) -> Dict[str, float]:
    return {"valid": x["valid"]}


def gen_relevance_acc(x: Dict[str, Any]) -> Dict[str, float]:
    # This function serves for both relevance and irrelevance tests, which share the exact opposite logic.
    # If `test_category` is "irrelevance", the model is expected to output no function call.
    # No function call means either the AST decoding fails (a error message is generated) or the decoded AST does not contain any function call (such as a empty list, `[]`).
    # If `test_category` is "relevance", the model is expected to output to a function call, and empty list doesn't count as a function call.
    acc = not x["contain_func_call"] if "irrelevance" in x["id"] else x["contain_func_call"]
    return {"valid": float(acc)}


class BFCLScoringFn(RegisteredBaseScoringFn):
    """
    A scoring_fn for BFCL
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.supported_fn_defs_registry = {
            bfcl.identifier: bfcl,
        }

    async def score_row(
        self,
        input_row: Dict[str, Any],
        scoring_fn_identifier: Optional[str] = "bfcl",
        scoring_params: Optional[ScoringFnParams] = None,
    ) -> ScoringResultRow:
        test_category = re.sub(r"_[0-9_-]+$", "", input_row["id"])
        score_result = postprocess(input_row, test_category)
        if test_category in {"irrelevance", "live_relevance", "live_irrelevance"}:
            score = gen_relevance_acc(score_result)["valid"]
        else:
            score = gen_valid(score_result)["valid"]
        return {
            "score": float(score),
        }
