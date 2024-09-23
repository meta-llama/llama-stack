# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

from typing import List, Optional

import fire
import httpx
from termcolor import cprint

from .shields import *  # noqa: F403


class ShieldsClient(Shields):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def list_shields(self) -> List[ShieldSpec]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/shields/list",
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return [ShieldSpec(**x) for x in response.json()]

    async def get_shield(self, shield_type: str) -> Optional[ShieldSpec]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/shields/get",
                params={
                    "shield_type": shield_type,
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            j = response.json()
            if j is None:
                return None

            return ShieldSpec(**j)


async def run_main(host: str, port: int, stream: bool):
    client = ShieldsClient(f"http://{host}:{port}")

    response = await client.list_shields()
    cprint(f"list_shields response={response}", "green")


def main(host: str, port: int, stream: bool = True):
    asyncio.run(run_main(host, port, stream))


if __name__ == "__main__":
    fire.Fire(main)
