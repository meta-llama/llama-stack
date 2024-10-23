# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import List

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.scoring import *  # noqa: F403
from llama_stack.apis.datasetio import *  # noqa: F403

from termcolor import cprint

from llama_stack.providers.datatypes import ScoringFunctionsProtocolPrivate

from .config import MetaReferenceScoringConfig


class MetaReferenceScoringImpl(Scoring, ScoringFunctionsProtocolPrivate):
    def __init__(
        self, config: MetaReferenceScoringConfig, datasetio_api: DatasetIO
    ) -> None:
        self.config = config
        self.datasetio_api = datasetio_api
        cprint(f"!!! MetaReferenceScoringImpl init {config} {datasetio_api}", "red")

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def score_batch(
        self, dataset_id: str, scoring_functions: List[str]
    ) -> ScoreBatchResponse:
        print("score_batch")

    async def score(
        self, input_rows: List[Dict[str, Any]], scoring_functions: List[str]
    ) -> ScoreResponse:
        print("score")
