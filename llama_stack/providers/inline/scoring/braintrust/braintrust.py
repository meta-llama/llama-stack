# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import List

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.scoring import *  # noqa: F403
from llama_stack.apis.scoring_functions import *  # noqa: F403
from llama_stack.apis.common.type_system import *  # noqa: F403
from llama_stack.apis.datasetio import *  # noqa: F403
from llama_stack.apis.datasets import *  # noqa: F403

import os

from autoevals.llm import Factuality
from autoevals.ragas import AnswerCorrectness

from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.datatypes import ScoringFunctionsProtocolPrivate

from llama_stack.providers.utils.scoring.aggregation_utils import aggregate_average

from .config import BraintrustScoringConfig
from .scoring_fn.fn_defs.answer_correctness import answer_correctness_fn_def
from .scoring_fn.fn_defs.factuality import factuality_fn_def


class BraintrustScoringImpl(
    Scoring, ScoringFunctionsProtocolPrivate, NeedsRequestProviderData
):
    def __init__(
        self,
        config: BraintrustScoringConfig,
        datasetio_api: DatasetIO,
        datasets_api: Datasets,
    ) -> None:
        self.config = config
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api

        self.braintrust_evaluators = {
            "braintrust::factuality": Factuality(),
            "braintrust::answer-correctness": AnswerCorrectness(),
        }
        self.supported_fn_defs_registry = {
            factuality_fn_def.identifier: factuality_fn_def,
            answer_correctness_fn_def.identifier: answer_correctness_fn_def,
        }

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def list_scoring_functions(self) -> List[ScoringFn]:
        scoring_fn_defs_list = [x for x in self.supported_fn_defs_registry.values()]
        for f in scoring_fn_defs_list:
            assert f.identifier.startswith(
                "braintrust"
            ), "All braintrust scoring fn must have identifier prefixed with 'braintrust'! "

        return scoring_fn_defs_list

    async def register_scoring_function(self, scoring_fn: ScoringFn) -> None:
        raise NotImplementedError(
            "Registering scoring function not allowed for braintrust provider"
        )

    async def validate_scoring_input_dataset_schema(self, dataset_id: str) -> None:
        dataset_def = await self.datasets_api.get_dataset(dataset_id=dataset_id)
        if not dataset_def.dataset_schema or len(dataset_def.dataset_schema) == 0:
            raise ValueError(
                f"Dataset {dataset_id} does not have a schema defined. Please define a schema for the dataset."
            )

        for required_column in ["generated_answer", "expected_answer", "input_query"]:
            if required_column not in dataset_def.dataset_schema:
                raise ValueError(
                    f"Dataset {dataset_id} does not have a '{required_column}' column."
                )
            if dataset_def.dataset_schema[required_column].type != "string":
                raise ValueError(
                    f"Dataset {dataset_id} does not have a '{required_column}' column of type 'string'."
                )

    async def set_api_key(self) -> None:
        # api key is in the request headers
        if not self.config.openai_api_key:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.openai_api_key:
                raise ValueError(
                    'Pass OpenAI API Key in the header X-LlamaStack-ProviderData as { "openai_api_key": <your api key>}'
                )
            self.config.openai_api_key = provider_data.openai_api_key

        os.environ["OPENAI_API_KEY"] = self.config.openai_api_key

    async def score_batch(
        self,
        dataset_id: str,
        scoring_functions: List[str],
        save_results_dataset: bool = False,
    ) -> ScoreBatchResponse:
        await self.set_api_key()
        await self.validate_scoring_input_dataset_schema(dataset_id=dataset_id)
        all_rows = await self.datasetio_api.get_rows_paginated(
            dataset_id=dataset_id,
            rows_in_page=-1,
        )
        res = await self.score(
            input_rows=all_rows.rows, scoring_functions=scoring_functions
        )
        if save_results_dataset:
            # TODO: persist and register dataset on to server for reading
            # self.datasets_api.register_dataset()
            raise NotImplementedError("Save results dataset not implemented yet")

        return ScoreBatchResponse(
            results=res.results,
        )

    async def score_row(
        self, input_row: Dict[str, Any], scoring_fn_identifier: Optional[str] = None
    ) -> ScoringResultRow:
        await self.set_api_key()
        assert scoring_fn_identifier is not None, "scoring_fn_identifier cannot be None"
        expected_answer = input_row["expected_answer"]
        generated_answer = input_row["generated_answer"]
        input_query = input_row["input_query"]
        evaluator = self.braintrust_evaluators[scoring_fn_identifier]

        result = evaluator(generated_answer, expected_answer, input=input_query)
        score = result.score
        return {"score": score, "metadata": result.metadata}

    async def score(
        self, input_rows: List[Dict[str, Any]], scoring_functions: List[str]
    ) -> ScoreResponse:
        await self.set_api_key()
        res = {}
        for scoring_fn_id in scoring_functions:
            if scoring_fn_id not in self.supported_fn_defs_registry:
                raise ValueError(f"Scoring function {scoring_fn_id} is not supported.")

            score_results = [
                await self.score_row(input_row, scoring_fn_id)
                for input_row in input_rows
            ]

            agg_results = aggregate_average(score_results)
            res[scoring_fn_id] = ScoringResult(
                score_rows=score_results,
                aggregated_results=agg_results,
            )

        return ScoreResponse(
            results=res,
        )
