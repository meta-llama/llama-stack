import asyncio
import json
from typing import AsyncGenerator

import fire
import httpx

from .api.endpoints import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChunk,
    CompletionRequest,
    InstructModel,
    ModelInference,
)


class ModelInferenceClient(ModelInference):
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
    client = ModelInferenceClient(f"http://{host}:{port}")

    message = UserMessage(content="hello world, help me out here")
    req = ChatCompletionRequest(
        model=InstructModel.llama3_70b_chat,
        messages=[message],
        stream=True,
    )
    async for event in client.chat_completion(
        ChatCompletionRequest(
            model=InstructModel.llama3_70b_chat,
            messages=[message],
            stream=True,
        )
    ):
        print(event)


def main(host: str, port: int):
    asyncio.run(run_main(host, port))


if __name__ == "__main__":
    fire.Fire(main)
