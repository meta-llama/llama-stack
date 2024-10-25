# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import os
from pathlib import Path

import fire
import httpx
from termcolor import cprint

from llama_stack.apis.datasets import *  # noqa: F403
from llama_stack.apis.scoring import *  # noqa: F403
from llama_stack.apis.common.type_system import *  # noqa: F403
from llama_stack.apis.datasetio.client import DatasetIOClient
from llama_stack.apis.datasets.client import DatasetsClient
from llama_stack.providers.tests.datasetio.test_datasetio import data_url_from_file


class ScoringClient(Scoring):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def score_batch(
        self, dataset_id: str, scoring_functions: List[str]
    ) -> ScoreBatchResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/scoring/score_batch",
                json={
                    "dataset_id": dataset_id,
                    "scoring_functions": scoring_functions,
                },
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            response.raise_for_status()
            if not response.json():
                return

            return ScoreBatchResponse(**response.json())

    async def score(
        self, input_rows: List[Dict[str, Any]], scoring_functions: List[str]
    ) -> ScoreResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/scoring/score",
                json={
                    "input_rows": input_rows,
                    "scoring_functions": scoring_functions,
                },
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            response.raise_for_status()
            if not response.json():
                return

            return ScoreResponse(**response.json())


async def run_main(host: str, port: int):
    client = DatasetsClient(f"http://{host}:{port}")

    # register dataset
    test_file = (
        Path(os.path.abspath(__file__)).parent.parent.parent
        / "providers/tests/datasetio/test_dataset.csv"
    )
    test_url = data_url_from_file(str(test_file))
    response = await client.register_dataset(
        DatasetDefWithProvider(
            identifier="test-dataset",
            provider_id="meta0",
            url=URL(
                uri=test_url,
            ),
            dataset_schema={
                "generated_answer": StringType(),
                "expected_answer": StringType(),
                "input_query": StringType(),
            },
        )
    )

    # list datasets
    list_dataset = await client.list_datasets()
    cprint(list_dataset, "blue")

    # datsetio client to get the rows
    datasetio_client = DatasetIOClient(f"http://{host}:{port}")
    response = await datasetio_client.get_rows_paginated(
        dataset_id="test-dataset",
        rows_in_page=4,
        page_token=None,
        filter_condition=None,
    )
    cprint(f"Returned {len(response.rows)} rows \n {response}", "green")

    # scoring client to score the rows
    scoring_client = ScoringClient(f"http://{host}:{port}")
    response = await scoring_client.score(
        input_rows=response.rows,
        scoring_functions=["equality"],
    )
    cprint(f"score response={response}", "blue")

    # test scoring batch using datasetio api
    scoring_client = ScoringClient(f"http://{host}:{port}")
    response = await scoring_client.score_batch(
        dataset_id="test-dataset",
        scoring_functions=["equality"],
    )
    cprint(f"score_batch response={response}", "cyan")


def main(host: str, port: int):
    asyncio.run(run_main(host, port))


if __name__ == "__main__":
    fire.Fire(main)
