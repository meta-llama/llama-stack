# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import fire
import httpx
from termcolor import cprint

from .datasets import *  # noqa: F403
from llama_stack.apis.datasets import *  # noqa: F403
from llama_stack.apis.common.type_system import *  # noqa: F403
from llama_stack.providers.tests.datasetio.test_datasetio import data_url_from_file


class DatasetsClient(Datasets):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_dataset(
        self,
        dataset_def: DatasetDefWithProvider,
    ) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/datasets/register",
                json={
                    "dataset_def": json.loads(dataset_def.json()),
                },
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            response.raise_for_status()
            return

    async def get_dataset(
        self,
        dataset_identifier: str,
    ) -> Optional[DatasetDefWithProvider]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/datasets/get",
                params={
                    "dataset_identifier": dataset_identifier,
                },
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            response.raise_for_status()
            if not response.json():
                return

            return DatasetDefWithProvider(**response.json())

    async def list_datasets(self) -> List[DatasetDefWithProvider]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/datasets/list",
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            response.raise_for_status()
            if not response.json():
                return

            return [DatasetDefWithProvider(**x) for x in response.json()]


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


def main(host: str, port: int):
    asyncio.run(run_main(host, port))


if __name__ == "__main__":
    fire.Fire(main)
