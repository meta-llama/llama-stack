# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
from typing import Any, AsyncGenerator

import fire
import httpx

from llama_stack.distribution.datatypes import RemoteProviderConfig
from pydantic import BaseModel
from termcolor import cprint

from .event_logger import EventLogger

from .inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChunk,
    CompletionRequest,
    Inference,
    UserMessage,
)


async def get_client_impl(config: RemoteProviderConfig, _deps: Any) -> Inference:
    return InferenceClient(config.url)


def encodable_dict(d: BaseModel):
    return json.loads(d.json())


class InferenceClient(Inference):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def completion(self, request: CompletionRequest) -> AsyncGenerator:
        raise NotImplementedError()

    async def chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/inference/chat_completion",
                json=encodable_dict(request),
                headers={"Content-Type": "application/json"},
                timeout=20,
            ) as response:
                if response.status_code != 200:
                    content = await response.aread()
                    cprint(
                        f"Error: HTTP {response.status_code} {content.decode()}", "red"
                    )
                    return

                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data = line[len("data: ") :]
                        try:
                            if request.stream:
                                if "error" in data:
                                    cprint(data, "red")
                                    continue

                                yield ChatCompletionResponseStreamChunk(
                                    **json.loads(data)
                                )
                            else:
                                yield ChatCompletionResponse(**json.loads(data))
                        except Exception as e:
                            print(data)
                            print(f"Error with parsing or validation: {e}")


async def run_main(host: str, port: int, stream: bool):
    client = InferenceClient(f"http://{host}:{port}")

    message = UserMessage(content="hello world, troll me in two-paragraphs about 42")
    cprint(f"User>{message.content}", "green")
    iterator = client.chat_completion(
        ChatCompletionRequest(
            model="Meta-Llama3.1-8B-Instruct",
            messages=[message],
            stream=stream,
        )
    )
    async for log in EventLogger().log(iterator):
        log.print()


def main(host: str, port: int, stream: bool = True):
    asyncio.run(run_main(host, port, stream))


if __name__ == "__main__":
    fire.Fire(main)
