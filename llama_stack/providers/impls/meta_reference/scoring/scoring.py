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

from llama_stack.providers.datatypes import ScoringFunctionsProtocolPrivate
from llama_stack.providers.impls.meta_reference.scoring.scorer.equality_scorer import (
    EqualityScorer,
)

from llama_stack.providers.impls.meta_reference.scoring.scorer.subset_of_scorer import (
    SubsetOfScorer,
)

from .config import MetaReferenceScoringConfig

SUPPORTED_SCORERS = [
    EqualityScorer,
    SubsetOfScorer,
]

SCORER_REGISTRY = {x.scoring_function_def.identifier: x for x in SUPPORTED_SCORERS}


class MetaReferenceScoringImpl(Scoring, ScoringFunctionsProtocolPrivate):
    def __init__(
        self,
        config: MetaReferenceScoringConfig,
        datasetio_api: DatasetIO,
        datasets_api: Datasets,
    ) -> None:
        self.config = config
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def list_scoring_functions(self) -> List[ScoringFunctionDef]:
        return [x.scoring_function_def for x in SUPPORTED_SCORERS]

    async def register_scoring_function(self, function_def: ScoringFunctionDef) -> None:
        raise NotImplementedError(
            "Dynamically registering scoring functions is not supported"
        )

    async def validate_scoring_input_dataset_schema(self, dataset_id: str) -> None:
        dataset_def = await self.datasets_api.get_dataset(dataset_identifier=dataset_id)
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

    async def score_batch(
        self,
        dataset_id: str,
        scoring_functions: List[str],
        save_results_dataset: bool = False,
    ) -> ScoreBatchResponse:
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

    async def score(
        self, input_rows: List[Dict[str, Any]], scoring_functions: List[str]
    ) -> ScoreResponse:
        res = {}
        for scoring_fn_id in scoring_functions:
            if scoring_fn_id not in SCORER_REGISTRY:
                raise ValueError(f"Scoring function {scoring_fn_id} is not supported.")
            scorer = SCORER_REGISTRY[scoring_fn_id]()
            score_results = scorer.score(input_rows)
            agg_results = scorer.aggregate(score_results)
            res[scoring_fn_id] = ScoringResult(
                score_rows=score_results,
                aggregated_results=agg_results,
            )

        return ScoreResponse(
            results=res,
        )
