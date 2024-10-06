# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import sys
from typing import Any, AsyncGenerator, List, Optional

import fire
import httpx

from llama_models.llama3.api.datatypes import ImageMedia, URL

from pydantic import BaseModel

from llama_models.llama3.api import *  # noqa: F403
from llama_stack.apis.inference import *  # noqa: F403
from termcolor import cprint

from llama_stack.distribution.datatypes import RemoteProviderConfig

from .event_logger import EventLogger


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

    async def chat_completion(
        self,
        model: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            tool_choice=tool_choice,
            tool_prompt_format=tool_prompt_format,
            stream=stream,
            logprobs=logprobs,
        )
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


async def run_main(host: str, port: int, stream: bool, model: Optional[str]):
    client = InferenceClient(f"http://{host}:{port}")

    if not model:
        model = "Llama3.1-8B-Instruct"

    message = UserMessage(
        content="hello world, write me a 2 sentence poem about the moon"
    )
    cprint(f"User>{message.content}", "green")
    iterator = client.chat_completion(
        model=model,
        messages=[message],
        stream=stream,
    )
    async for log in EventLogger().log(iterator):
        log.print()


async def run_mm_main(
    host: str, port: int, stream: bool, path: Optional[str], model: Optional[str]
):
    client = InferenceClient(f"http://{host}:{port}")

    if not model:
        model = "Llama3.2-11B-Vision-Instruct"

    message = UserMessage(
        content=[
            ImageMedia(image=URL(uri=f"file://{path}")),
            "Describe this image in two sentences",
        ],
    )
    cprint(f"User>{message.content}", "green")
    iterator = client.chat_completion(
        model=model,
        messages=[message],
        stream=stream,
    )
    async for log in EventLogger().log(iterator):
        log.print()


def main(
    host: str,
    port: int,
    stream: bool = True,
    mm: bool = False,
    file: Optional[str] = None,
    model: Optional[str] = None,
):
    if mm:
        asyncio.run(run_mm_main(host, port, stream, file, model))
    else:
        asyncio.run(run_main(host, port, stream, model))


if __name__ == "__main__":
    fire.Fire(main)
