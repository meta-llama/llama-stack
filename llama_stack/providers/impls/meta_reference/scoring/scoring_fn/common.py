# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from llama_stack.apis.scoring import ScoringResultRow

FN_DEFS_PATH = Path(__file__).parent / "fn_defs"


def aggregate_accuracy(scoring_results: List[ScoringResultRow]) -> Dict[str, Any]:
    num_correct = sum(result["score"] for result in scoring_results)
    avg_score = num_correct / len(scoring_results)

    return {
        "accuracy": avg_score,
        "num_correct": num_correct,
        "num_total": len(scoring_results),
    }


def aggregate_average(scoring_results: List[ScoringResultRow]) -> Dict[str, Any]:
    return {
        "average": sum(
            result["score"] for result in scoring_results if result["score"] is not None
        )
        / len([_ for _ in scoring_results if _["score"] is not None]),
    }


def aggregate_count(scoring_results: List[ScoringResultRow]) -> Dict[str, Any]:
    result = Counter()
    for x in scoring_results:
        if x["grade"] is None:
            result["None"] += 1
        else:
            result[x["grade"]] += 1
    return dict(result)
