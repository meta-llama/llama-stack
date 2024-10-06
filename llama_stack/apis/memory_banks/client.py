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

from .memory_banks import *  # noqa: F403


class MemoryBanksClient(MemoryBanks):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def list_available_memory_banks(self) -> List[MemoryBankSpec]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/memory_banks/list",
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return [MemoryBankSpec(**x) for x in response.json()]

    async def get_serving_memory_bank(
        self, bank_type: MemoryBankType
    ) -> Optional[MemoryBankSpec]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/memory_banks/get",
                params={
                    "bank_type": bank_type.value,
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            j = response.json()
            if j is None:
                return None
            return MemoryBankSpec(**j)


async def run_main(host: str, port: int, stream: bool):
    client = MemoryBanksClient(f"http://{host}:{port}")

    response = await client.list_available_memory_banks()
    cprint(f"list_memory_banks response={response}", "green")


def main(host: str, port: int, stream: bool = True):
    asyncio.run(run_main(host, port, stream))


if __name__ == "__main__":
    fire.Fire(main)
