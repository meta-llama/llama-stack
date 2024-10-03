# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

from typing import List

import fire
import httpx
from termcolor import cprint

from .inspect import *  # noqa: F403


class InspectClient(Inspect):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def list_providers(self) -> Dict[str, ProviderInfo]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/providers/list",
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            print(response.json())
            return {
                k: [ProviderInfo(**vi) for vi in v] for k, v in response.json().items()
            }

    async def list_routes(self) -> Dict[str, List[RouteInfo]]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/routes/list",
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return {
                k: [RouteInfo(**vi) for vi in v] for k, v in response.json().items()
            }

    async def health(self) -> HealthInfo:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/health",
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            j = response.json()
            if j is None:
                return None
            return HealthInfo(**j)


async def run_main(host: str, port: int):
    client = InspectClient(f"http://{host}:{port}")

    response = await client.list_providers()
    cprint(f"list_providers response={response}", "green")

    response = await client.list_routes()
    cprint(f"list_routes response={response}", "blue")

    response = await client.health()
    cprint(f"health response={response}", "yellow")


def main(host: str, port: int):
    asyncio.run(run_main(host, port))


if __name__ == "__main__":
    fire.Fire(main)
