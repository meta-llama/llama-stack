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
from llama_stack.apis.inference.inference import Inference
from llama_stack.providers.datatypes import ScoringFunctionsProtocolPrivate
from llama_stack.providers.impls.meta_reference.scoring.scoring_fn.equality_scoring_fn import (
    EqualityScoringFn,
)

from llama_stack.providers.impls.meta_reference.scoring.scoring_fn.llm_as_judge_scoring_fn import (
    LlmAsJudgeScoringFn,
)

from llama_stack.providers.impls.meta_reference.scoring.scoring_fn.subset_of_scoring_fn import (
    SubsetOfScoringFn,
)

from .config import MetaReferenceScoringConfig

FIXED_FNS = [EqualityScoringFn, SubsetOfScoringFn]

LLM_JUDGE_FNS = [LlmAsJudgeScoringFn]


class MetaReferenceScoringImpl(Scoring, ScoringFunctionsProtocolPrivate):
    def __init__(
        self,
        config: MetaReferenceScoringConfig,
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
        for x in FIXED_FNS:
            impl = x()
            for fn_defs in impl.get_supported_scoring_fn_defs():
                self.scoring_fn_id_impls[fn_defs.identifier] = impl
        for x in LLM_JUDGE_FNS:
            impl = x(inference_api=self.inference_api)
            for fn_defs in impl.get_supported_scoring_fn_defs():
                self.scoring_fn_id_impls[fn_defs.identifier] = impl
                self.llm_as_judge_fn = impl

    async def shutdown(self) -> None: ...

    async def list_scoring_functions(self) -> List[ScoringFnDef]:
        scoring_fn_defs_list = [
            fn_def
            for impl in self.scoring_fn_id_impls.values()
            for fn_def in impl.get_supported_scoring_fn_defs()
        ]

        for f in scoring_fn_defs_list:
            assert f.identifier.startswith(
                "meta-reference"
            ), "All meta-reference scoring fn must have identifier prefixed with 'meta-reference'! "

        return scoring_fn_defs_list

    async def register_scoring_function(self, function_def: ScoringFnDef) -> None:
        self.llm_as_judge_fn.register_scoring_fn_def(function_def)
        self.scoring_fn_id_impls[function_def.identifier] = self.llm_as_judge_fn

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
            if scoring_fn_id not in self.scoring_fn_id_impls:
                raise ValueError(f"Scoring function {scoring_fn_id} is not supported.")
            scoring_fn = self.scoring_fn_id_impls[scoring_fn_id]
            score_results = await scoring_fn.score(input_rows, scoring_fn_id)
            agg_results = await scoring_fn.aggregate(score_results)
            res[scoring_fn_id] = ScoringResult(
                score_rows=score_results,
                aggregated_results=agg_results,
            )

        return ScoreResponse(
            results=res,
        )
