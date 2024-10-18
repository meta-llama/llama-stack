# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json

from typing import Any, Dict, List, Optional

import fire
import httpx
from termcolor import cprint

from .memory_banks import *  # noqa: F403


def deserialize_memory_bank_def(
    j: Optional[Dict[str, Any]]
) -> MemoryBankDefWithProvider:
    if j is None:
        return None

    if "type" not in j:
        raise ValueError("Memory bank type not specified")
    type = j["type"]
    if type == MemoryBankType.vector.value:
        return VectorMemoryBankDef(**j)
    elif type == MemoryBankType.keyvalue.value:
        return KeyValueMemoryBankDef(**j)
    elif type == MemoryBankType.keyword.value:
        return KeywordMemoryBankDef(**j)
    elif type == MemoryBankType.graph.value:
        return GraphMemoryBankDef(**j)
    else:
        raise ValueError(f"Unknown memory bank type: {type}")


class MemoryBanksClient(MemoryBanks):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def list_memory_banks(self) -> List[MemoryBankDefWithProvider]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/memory_banks/list",
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return [deserialize_memory_bank_def(x) for x in response.json()]

    async def register_memory_bank(
        self, memory_bank: MemoryBankDefWithProvider
    ) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/memory_banks/register",
                json={
                    "memory_bank": json.loads(memory_bank.json()),
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

    async def get_memory_bank(
        self,
        identifier: str,
    ) -> Optional[MemoryBankDefWithProvider]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/memory_banks/get",
                params={
                    "identifier": identifier,
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            j = response.json()
            return deserialize_memory_bank_def(j)


async def run_main(host: str, port: int, stream: bool):
    client = MemoryBanksClient(f"http://{host}:{port}")

    response = await client.list_memory_banks()
    cprint(f"list_memory_banks response={response}", "green")

    # register memory bank for the first time
    response = await client.register_memory_bank(
        VectorMemoryBankDef(
            identifier="test_bank2",
            embedding_model="all-MiniLM-L6-v2",
            chunk_size_in_tokens=512,
            overlap_size_in_tokens=64,
        )
    )
    cprint(f"register_memory_bank response={response}", "blue")

    # list again after registering
    response = await client.list_memory_banks()
    cprint(f"list_memory_banks response={response}", "green")


def main(host: str, port: int, stream: bool = True):
    asyncio.run(run_main(host, port, stream))


if __name__ == "__main__":
    fire.Fire(main)
