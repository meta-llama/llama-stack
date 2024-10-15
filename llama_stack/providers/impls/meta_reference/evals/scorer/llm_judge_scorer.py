# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio
import threading

import numpy as np

from llama_stack.distribution.registry.generator_processors import (
    GeneratorProcessorRegistry,
)
from llama_stack.providers.impls.meta_reference.evals.generator.inference_generator import (
    InferenceGenerator,
)

from llama_stack.apis.evals.evals import *  # noqa: F401 F403
from llama_stack.apis.datasets.datasets import *  # noqa: F401 F403
from llama_stack.apis.inference import *  # noqa: F403


class LlamaStackLLMJudgeScorer(BaseScorer[ScorerInputSample]):
    def __init__(self, llm_judge_config: LLMJudgeConfig, inference_api: Inference):
        self.llm_judge_config = llm_judge_config
        self.inference_api = inference_api
        # https://stackoverflow.com/questions/74703727/how-to-call-async-function-from-sync-funcion-and-get-result-while-a-loop-is-alr
        # We will use another thread wih its own event loop to run the async api within sync function
        self._loop = asyncio.new_event_loop()
        self._thr = threading.Thread(
            target=self._loop.run_forever, name="Async Runner", daemon=True
        )
        if not self._thr.is_alive():
            self._thr.start()

    def score_sample(self, scorer_input_sample: ScorerInputSample) -> SingleEvalResult:
        input_query = scorer_input_sample.input_query
        generated_answer = scorer_input_sample.generated_answer
        expected_answer = scorer_input_sample.expected_answer

        # Judge F1
        processor = GeneratorProcessorRegistry.get(
            self.llm_judge_config.judge_processor_config.processor_identifier
        )()
        data_sample = DictSample(
            data={
                "input_query": input_query,
                "generated_answer": generated_answer,
                "expected_answer": expected_answer,
            }
        )
        preprocessed_sample = processor.preprocess_sample(data_sample)

        # Judge Generation
        generator = InferenceGenerator(
            model=self.llm_judge_config.judge_model_generation_config.model,
            inference_api=self.inference_api,
        )

        future = asyncio.run_coroutine_threadsafe(
            generator.generate([preprocessed_sample]), self._loop
        )
        generation_outputs = future.result()
        # Judge F2
        postprocessed_sample = processor.postprocess_sample(
            generation_outputs[0], data_sample
        )

        # Judge F3
        score = float(postprocessed_sample.generated_answer)

        return SingleEvalResult(score_data={"judge_score": score})

    def aggregate_results(self, eval_results: List[SingleEvalResult]) -> EvalResult:
        avg_score = np.average(
            [result.score_data["judge_score"] for result in eval_results]
        )

        return EvalResult(
            metrics={
                "avg_judge_score": avg_score,
            }
        )
