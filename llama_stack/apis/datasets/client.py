# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
from typing import Optional

import fire
import httpx
from termcolor import cprint

from .datasets import *  # noqa: F403


def deserialize_dataset_def(j: Optional[Dict[str, Any]]) -> Optional[DatasetDef]:
    if not j:
        return None
    if j["type"] == "huggingface":
        return HuggingfaceDatasetDef(**j)
    elif j["type"] == "custom":
        return CustomDatasetDef(**j)
    else:
        raise ValueError(f"Unknown dataset type: {j['type']}")


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
    ) -> CreateDatasetResponse:
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
            return CreateDatasetResponse(**response.json())

    async def get_dataset(
        self,
        dataset_identifier: str,
    ) -> Optional[DatasetDef]:
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

            return deserialize_dataset_def(response.json())

    async def delete_dataset(
        self,
        dataset_identifier: str,
    ) -> DeleteDatasetResponse:
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
            return DeleteDatasetResponse(**response.json())

    async def list_dataset(
        self,
    ) -> List[DatasetDef]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/datasets/list",
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            response.raise_for_status()
            if not response.json():
                return

            return [deserialize_dataset_def(x) for x in response.json()]


async def run_main(host: str, port: int):
    client = DatasetClient(f"http://{host}:{port}")

    # register dataset
    response = await client.create_dataset(
        dataset_def=CustomDatasetDef(
            identifier="test-dataset",
            url="https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv",
        ),
    )
    cprint(response, "green")

    # get dataset
    get_dataset = await client.get_dataset(
        dataset_identifier="test-dataset",
    )
    cprint(get_dataset, "cyan")

    # delete dataset
    delete_dataset = await client.delete_dataset(
        dataset_identifier="test-dataset",
    )
    cprint(delete_dataset, "red")

    # get again after deletion
    get_dataset = await client.get_dataset(
        dataset_identifier="test-dataset",
    )
    cprint(get_dataset, "yellow")

    # list datasets
    list_dataset = await client.list_dataset()
    cprint(list_dataset, "blue")


def main(host: str, port: int):
    asyncio.run(run_main(host, port))


if __name__ == "__main__":
    fire.Fire(main)
