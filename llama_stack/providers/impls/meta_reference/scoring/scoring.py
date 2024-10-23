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

from termcolor import cprint

from llama_stack.providers.datatypes import ScoringFunctionsProtocolPrivate
from llama_stack.providers.impls.meta_reference.scoring.scorer.equality_scorer import (
    EqualityScorer,
)

from .config import MetaReferenceScoringConfig

SUPPORTED_SCORERS = [
    EqualityScorer,
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
        cprint(f"!!! MetaReferenceScoringImpl init {config} {datasets_api}", "red")

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def list_scoring_functions(self) -> List[ScoringFunctionDef]:
        return [x.scoring_function_def for x in SUPPORTED_SCORERS]

    async def register_scoring_function(self, function_def: ScoringFunctionDef) -> None:
        raise NotImplementedError(
            "Dynamically registering scoring functions is not supported"
        )

    async def score_batch(
        self,
        dataset_id: str,
        scoring_functions: List[str],
        save_results_dataset: bool = False,
    ) -> ScoreBatchResponse:
        rows_paginated = await self.datasetio_api.get_rows_paginated(
            dataset_id=dataset_id,
            rows_in_page=-1,
        )
        res = await self.score(
            input_rows=rows_paginated.rows, scoring_functions=scoring_functions
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
            scorer = SCORER_REGISTRY[scoring_fn_id]()
            score_results = scorer.score(input_rows)
            agg_results = scorer.aggregate(score_results)
            res[scoring_fn_id] = agg_results

        return ScoreResponse(
            results=res,
        )
