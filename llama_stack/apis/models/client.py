# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json

from typing import List, Optional

import fire
import httpx
from termcolor import cprint

from .models import *  # noqa: F403


class ModelsClient(Models):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def list_models(self) -> List[ModelDefWithProvider]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/models/list",
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return [ModelDefWithProvider(**x) for x in response.json()]

    async def register_model(self, model: ModelDefWithProvider) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/models/register",
                json={
                    "model": json.loads(model.json()),
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

    async def get_model(self, identifier: str) -> Optional[ModelDefWithProvider]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/models/get",
                params={
                    "identifier": identifier,
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            j = response.json()
            if j is None:
                return None
            return ModelDefWithProvider(**j)


async def run_main(host: str, port: int, stream: bool):
    client = ModelsClient(f"http://{host}:{port}")

    response = await client.list_models()
    cprint(f"list_models response={response}", "green")

    response = await client.get_model("Llama3.1-8B-Instruct")
    cprint(f"get_model response={response}", "blue")

    response = await client.get_model("Llama-Guard-3-1B")
    cprint(f"get_model response={response}", "red")


def main(host: str, port: int, stream: bool = True):
    asyncio.run(run_main(host, port, stream))


if __name__ == "__main__":
    fire.Fire(main)
