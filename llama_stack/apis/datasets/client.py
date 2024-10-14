# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json

import fire
import httpx

from .datasets import *  # noqa: F403


class DatasetClient(Datasets):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def create_dataset(
        self,
        dataset_def: DatasetDef,
    ) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/datasets/create",
                json={
                    "dataset_def": json.loads(dataset_def.json()),
                },
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            response.raise_for_status()
            return None

    async def get_dataset(
        self,
        dataset_identifier: str,
    ) -> DatasetDef:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/datasets/create",
                json={
                    "dataset_identifier": dataset_identifier,
                },
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            response.raise_for_status()
            return DatasetDef(**response.json())

    async def delete_dataset(
        self,
        dataset_identifier: str,
    ) -> DatasetDef:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/datasets/delete",
                json={
                    "dataset_identifier": dataset_identifier,
                },
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            response.raise_for_status()
            return None


async def run_main(host: str, port: int):
    client = DatasetClient(f"http://{host}:{port}")

    # Custom Eval Task
    response = await client.create_dataset(
        dataset_def=CustomDatasetDef(
            identifier="test-dataset",
            url="https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv",
        ),
    )


def main(host: str, port: int):
    asyncio.run(run_main(host, port))


if __name__ == "__main__":
    fire.Fire(main)
