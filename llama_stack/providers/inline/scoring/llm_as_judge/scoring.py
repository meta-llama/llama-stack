# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any, Dict, List, Optional

from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.inference.inference import Inference

from llama_stack.apis.scoring import (
    ScoreBatchResponse,
    ScoreResponse,
    Scoring,
    ScoringResult,
)
from llama_stack.apis.scoring_functions import ScoringFn, ScoringFnParams
from llama_stack.providers.datatypes import ScoringFunctionsProtocolPrivate
from llama_stack.providers.utils.common.data_schema_validator_mixin import (
    DataSchemaValidatorMixin,
)

from .config import LlmAsJudgeScoringConfig
from .scoring_fn.llm_as_judge_scoring_fn import LlmAsJudgeScoringFn


LLM_JUDGE_FNS = [LlmAsJudgeScoringFn]


class LlmAsJudgeScoringImpl(
    Scoring, ScoringFunctionsProtocolPrivate, DataSchemaValidatorMixin
):
    def __init__(
        self,
        config: LlmAsJudgeScoringConfig,
        datasetio_api: DatasetIO,
        datasets_api: Datasets,
        inference_api: Inference,
    ) -> None:
        self.config = config
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api
        self.inference_api = inference_api
        self.scoring_fn_id_impls = {}

    async def initialize(self) -> None:
        for fn in LLM_JUDGE_FNS:
            impl = fn(inference_api=self.inference_api)
            for fn_defs in impl.get_supported_scoring_fn_defs():
                self.scoring_fn_id_impls[fn_defs.identifier] = impl
                self.llm_as_judge_fn = impl

    async def shutdown(self) -> None: ...

    async def list_scoring_functions(self) -> List[ScoringFn]:
        scoring_fn_defs_list = [
            fn_def
            for impl in self.scoring_fn_id_impls.values()
            for fn_def in impl.get_supported_scoring_fn_defs()
        ]

        for f in scoring_fn_defs_list:
            assert f.identifier.startswith(
                "llm-as-judge"
            ), "All llm-as-judge scoring fn must have identifier prefixed with 'llm-as-judge'! "

        return scoring_fn_defs_list

    async def register_scoring_function(self, function_def: ScoringFn) -> None:
        raise NotImplementedError("Register scoring function not implemented yet")

    async def score_batch(
        self,
        dataset_id: str,
        scoring_functions: Dict[str, Optional[ScoringFnParams]] = None,
        save_results_dataset: bool = False,
    ) -> ScoreBatchResponse:
        dataset_def = await self.datasets_api.get_dataset(dataset_id=dataset_id)
        self.validate_dataset_schema_for_scoring(dataset_def.dataset_schema)

        all_rows = await self.datasetio_api.get_rows_paginated(
            dataset_id=dataset_id,
            rows_in_page=-1,
        )
        res = await self.score(
            input_rows=all_rows.rows,
            scoring_functions=scoring_functions,
        )
        if save_results_dataset:
            # TODO: persist and register dataset on to server for reading
            # self.datasets_api.register_dataset()
            raise NotImplementedError("Save results dataset not implemented yet")

        return ScoreBatchResponse(
            results=res.results,
        )

    async def score(
        self,
        input_rows: List[Dict[str, Any]],
        scoring_functions: Dict[str, Optional[ScoringFnParams]] = None,
    ) -> ScoreResponse:
        res = {}
        for scoring_fn_id in scoring_functions.keys():
            if scoring_fn_id not in self.scoring_fn_id_impls:
                raise ValueError(f"Scoring function {scoring_fn_id} is not supported.")
            scoring_fn = self.scoring_fn_id_impls[scoring_fn_id]
            scoring_fn_params = scoring_functions.get(scoring_fn_id, None)
            score_results = await scoring_fn.score(
                input_rows, scoring_fn_id, scoring_fn_params
            )
            agg_results = await scoring_fn.aggregate(
                score_results, scoring_fn_id, scoring_fn_params
            )
            res[scoring_fn_id] = ScoringResult(
                score_rows=score_results,
                aggregated_results=agg_results,
            )

        return ScoreResponse(
            results=res,
        )
