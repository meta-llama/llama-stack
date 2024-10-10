# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import os
from pathlib import Path

from typing import Any, Dict, List, Optional

import fire
import httpx

from llama_stack.distribution.datatypes import RemoteProviderConfig

from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.apis.memory_banks.client import MemoryBanksClient
from llama_stack.providers.utils.memory.file_utils import data_url_from_file


async def get_client_impl(config: RemoteProviderConfig, _deps: Any) -> Memory:
    return MemoryClient(config.url)


class MemoryClient(Memory):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def insert_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
    ) -> None:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{self.base_url}/memory/insert",
                json={
                    "bank_id": bank_id,
                    "documents": [d.dict() for d in documents],
                },
                headers={"Content-Type": "application/json"},
                timeout=20,
            )
            r.raise_for_status()

    async def query_documents(
        self,
        bank_id: str,
        query: InterleavedTextMedia,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryDocumentsResponse:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{self.base_url}/memory/query",
                json={
                    "bank_id": bank_id,
                    "query": query,
                    "params": params,
                },
                headers={"Content-Type": "application/json"},
                timeout=20,
            )
            r.raise_for_status()
            return QueryDocumentsResponse(**r.json())


async def run_main(host: str, port: int, stream: bool):
    banks_client = MemoryBanksClient(f"http://{host}:{port}")

    bank = VectorMemoryBankDef(
        identifier="test_bank",
        provider_id="",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size_in_tokens=512,
        overlap_size_in_tokens=64,
    )
    await banks_client.register_memory_bank(bank)

    retrieved_bank = await banks_client.get_memory_bank(bank.identifier)
    assert retrieved_bank is not None
    assert retrieved_bank.embedding_model == "all-MiniLM-L6-v2"

    urls = [
        "memory_optimizations.rst",
        "chat.rst",
        "llama3.rst",
        "datasets.rst",
        "qat_finetune.rst",
        "lora_finetune.rst",
    ]
    documents = [
        MemoryBankDocument(
            document_id=f"num-{i}",
            content=URL(
                uri=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}"
            ),
            mime_type="text/plain",
        )
        for i, url in enumerate(urls)
    ]

    this_dir = os.path.dirname(__file__)
    files = [Path(this_dir).parent.parent.parent / "CONTRIBUTING.md"]
    documents += [
        MemoryBankDocument(
            document_id=f"num-{i}",
            content=data_url_from_file(path),
        )
        for i, path in enumerate(files)
    ]

    client = MemoryClient(f"http://{host}:{port}")

    # insert some documents
    await client.insert_documents(
        bank_id=bank.identifier,
        documents=documents,
    )

    # query the documents
    response = await client.query_documents(
        bank_id=bank.identifier,
        query=[
            "How do I use Lora?",
        ],
    )
    for chunk, score in zip(response.chunks, response.scores):
        print(f"Score: {score}")
        print(f"Chunk:\n========\n{chunk}\n========\n")

    response = await client.query_documents(
        bank_id=bank.identifier,
        query=[
            "Tell me more about llama3 and torchtune",
        ],
    )
    for chunk, score in zip(response.chunks, response.scores):
        print(f"Score: {score}")
        print(f"Chunk:\n========\n{chunk}\n========\n")


def main(host: str, port: int, stream: bool = True):
    asyncio.run(run_main(host, port, stream))


if __name__ == "__main__":
    fire.Fire(main)
