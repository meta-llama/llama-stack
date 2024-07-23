# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
from termcolor import cprint
from typing import AsyncGenerator

from urllib.request import getproxies

import fire
import httpx

from .api import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChunk,
    CompletionRequest,
    Inference,
    InstructModel,
    UserMessage,
)
from .event_logger import EventLogger

print(getproxies())
# import sys
# sys.exit(0)


class InferenceClient(Inference):
    def __init__(self, base_url: str):
        print(f"Initializing client for {base_url}")
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
                data=request.json(),
                headers={"Content-Type": "application/json"},
                timeout=20,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data = line[len("data: ") :]
                        try:
                            yield ChatCompletionResponseStreamChunk(**json.loads(data))
                        except Exception as e:
                            print(data)
                            print(f"Error with parsing or validation: {e}")


async def run_main(host: str, port: int):
    client = InferenceClient(f"http://{host}:{port}")

    message = UserMessage(content="hello world, help me out here")
    cprint(f"User>{message.content}", "green")
    req = ChatCompletionRequest(
        model=InstructModel.llama3_70b_chat,
        messages=[message],
        stream=True,
    )
    iterator = client.chat_completion(
        ChatCompletionRequest(
            model=InstructModel.llama3_8b_chat,
            messages=[message],
            stream=True,
        )
    )
    async for log in EventLogger().log(iterator):
        log.print()


def main(host: str, port: int):
    asyncio.run(run_main(host, port))


if __name__ == "__main__":
    fire.Fire(main)
