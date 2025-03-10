# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import re
from typing import Any, Dict, Optional

from llama_stack.apis.scoring import ScoringResultRow
from llama_stack.apis.scoring_functions import ScoringFnParams, ScoringFnParamsType
from llama_stack.providers.utils.scoring.base_scoring_fn import RegisteredBaseScoringFn

from .fn_defs.regex_parser_math_response import (
    regex_parser_math_response,
)
from ..utils.math_utils import normalize_final_answer, first_answer, try_evaluate_frac, try_evaluate_latex, extract_result_from_boxed


class RegexParserMathResponseScoringFn(RegisteredBaseScoringFn):
    """
    A scoring_fn for math benchamrks that parses answer from generated response according to context and check match with expected_answer.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.supported_fn_defs_registry = {
            regex_parser_math_response.identifier: regex_parser_math_response,
        }

    async def score_row(
        self,
        input_row: Dict[str, Any],
        scoring_fn_identifier: Optional[str] = None,
        scoring_params: Optional[ScoringFnParams] = None,
    ) -> ScoringResultRow:
        print('I reach RegexParserMathResponseScoringFn')
        assert scoring_fn_identifier is not None, "Scoring function identifier not found."
        fn_def = self.supported_fn_defs_registry[scoring_fn_identifier]
        if scoring_params is not None:
            fn_def.params = scoring_params

        assert fn_def.params is not None and fn_def.params.type == ScoringFnParamsType.regex_parser.value, (
            f"RegexParserScoringFnParams not found for {fn_def}."
        )

        expected_answer = input_row["expected_answer"]
        expected_answer = r"""
        We have that $r = \sqrt{0^2 + 3^2} = 3.$ Also, if we draw the line connecting the origin and $(0,3),$ this line makes an angle of $\frac{\pi}{2}$ with the positive $x$-axis.

        [asy]
        unitsize(0.8 cm);

        draw((-0.5,0)--(3.5,0));
        draw((0,-0.5)--(0,3.5));
        draw(arc((0,0),3,0,90),red,Arrow(6));

        dot((0,3), red);
        label("$(0,3)$", (0,3), W);
        dot((3,0), red);
        [/asy]

        Therefore, the polar coordinates are $\boxed{\left( 3, \frac{\pi}{2} \right)}.$
        """
        generated_answer = input_row["generated_answer"]

        print('expected_answer', expected_answer)
        print('generated_answer', generated_answer)

        parsing_regexes = fn_def.params.parsing_regexes

        assert len(parsing_regexes) == 1, "Only one parsing regex is supported for regex_parser_math_response scoring function."

        parsing_regexes = fn_def.params.parsing_regexes[0]

        print('parsing_regexes', parsing_regexes)

        
        normalized_generated_answer = normalize_final_answer(
                first_answer(generated_answer),
                parsing_regexes,
                match_first=True,
            )
        print('normalized_generated_answer_1', normalized_generated_answer)
        

        normalized_generated_answer = try_evaluate_frac(try_evaluate_latex(normalized_generated_answer))

        print('normalized_generated_answer_2', normalized_generated_answer)


        # print('extract_result_from_boxed', extract_result_from_boxed(expected_answer)) 
        # normalized_expected_answer = normalize_final_answer(extract_result_from_boxed(expected_answer), r".*final answer is:?\s*\$\\boxed{(?P<X>.*)}\$")
        normalized_expected_answer = normalize_final_answer(expected_answer, r"\$\\boxed{(?P<X>.*)}\$")
        print('normalized_expected_answer_1', normalized_expected_answer)
        normalized_expected_answer = try_evaluate_frac(try_evaluate_latex(normalized_expected_answer))

        print('normalized_expected_answer_2', normalized_expected_answer)

        score = 1.0 if normalized_generated_answer == normalized_expected_answer else 0.0
        return {
            "score": score,
        }
