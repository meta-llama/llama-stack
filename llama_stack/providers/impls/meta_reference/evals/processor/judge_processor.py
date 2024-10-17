# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import re

from llama_stack.apis.evals import *  # noqa: F403

JUDGE_PROMPT = """
You will be given a question, a expected_answer, and a system_answer.
Your task is to provide a 'total rating' scoring how well the system_answer answers compared with ground truth in expected_answer in terms of factual correctness to the question.
Give your answer as a integer on a scale of 0 to 5, where 0 means that the system_answer is not correct at all compared with expected_answer, and 5 means that the answer completely and correctly answers the question.

Provide your feedback as follows:

Feedback:::
Total rating: (your rating, as a int between 0 and 5)

Now here are the question, expected_answer, system_answer.

Question: {question}
Expected Answer: {expected_answer}
System Answer: {answer}

Feedback:::
Total rating:
"""


class JudgeProcessor(
    BaseGeneratorProcessor[
        DictSample, PreprocessedSample, GenerationResponseSample, ScorerInputSample
    ]
):
    """
    Generator processor for LLM Judge
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess_sample(self, sample: DictSample) -> PreprocessedSample:
        content = JUDGE_PROMPT.format(
            question=sample.data["input_query"],
            expected_answer=sample.data["expected_answer"],
            answer=sample.data["generated_answer"],
        )
        preprocessed_msgs = [
            {
                "role": "user",
                "content": content,
            }
        ]
        processed_sample = PreprocessedSample(
            generation_input=GenerationInput(
                messages=preprocessed_msgs,
            )
        )
        return processed_sample

    def postprocess_sample(
        self, generation_sample: GenerationResponseSample, dataset_sample: DictSample
    ) -> ScorerInputSample:
        response_text = generation_sample.generation_output.completion_message
        match = re.search(r"Total rating: (\d+)", response_text)
        judge_rating = int(match.group(1))

        return ScorerInputSample(
            generated_answer=str(judge_rating),
            expected_answer=dataset_sample.data["expected_answer"],
            generation_output=PostprocessedGeneration(
                completion_message=response_text,
            ),
        )
