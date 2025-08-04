# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Fake response generation for testing inference providers without making real API calls.
"""

import asyncio
import os
import time
from collections.abc import AsyncGenerator
from typing import Any
from openai.types.chat import ChatCompletion

from pydantic import BaseModel


class FakeConfig(BaseModel):
    response_length: int = 100
    latency_ms: int = 50


def parse_fake_config() -> FakeConfig:
    """Parse fake mode configuration from environment variable."""
    mode_str = os.environ.get("LLAMA_STACK_TEST_INFERENCE_MODE", "live").lower()
    config = {}

    if ":" in mode_str:
        parts = mode_str.split(":")
        for part in parts[1:]:
            if "=" in part:
                key, value = part.split("=", 1)
                config[key] = int(value)
    return FakeConfig(**config)


def generate_fake_content(word_count: int) -> str:
    """Generate fake response content with specified word count."""
    words = [
        "This",
        "is",
        "a",
        "synthetic",
        "response",
        "generated",
        "for",
        "testing",
        "purposes",
        "only",
        "The",
        "content",
        "simulates",
        "realistic",
        "language",
        "model",
        "output",
        "patterns",
        "and",
        "structures",
        "It",
        "includes",
        "various",
        "sentence",
        "types",
        "and",
        "maintains",
        "coherent",
        "flow",
        "throughout",
        "These",
        "responses",
        "help",
        "test",
        "system",
        "performance",
        "without",
        "requiring",
        "real",
        "model",
        "calls",
    ]

    return " ".join(words[i % len(words)] for i in range(word_count)) + "."


def generate_fake_chat_completion(body: dict[str, Any], config: FakeConfig) -> Any:
    """Generate fake OpenAI chat completion response."""
    model = body.get("model", "gpt-3.5-turbo")
    messages = body.get("messages", [])

    # Calculate fake token counts based on input
    prompt_tokens = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            prompt_tokens += len(content.split())
        elif isinstance(content, list):
            # Handle content arrays (images, etc.)
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    prompt_tokens += len(item.get("text", "").split())

    response_length = config.response_length
    fake_content = generate_fake_content(response_length)
    completion_tokens = len(fake_content.split())

    response_data = {
        "id": f"chatcmpl-fake-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": fake_content,
                    "function_call": None,
                    "tool_calls": None,
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "system_fingerprint": None,
    }
    time.sleep(config.latency_ms / 1000.0)

    return ChatCompletion.model_validate(response_data)


def generate_fake_completion(body: dict[str, Any], config: FakeConfig) -> dict[str, Any]:
    """Generate fake OpenAI completion response."""
    raise NotImplementedError("Fake completions not implemented yet")


def generate_fake_embeddings(body: dict[str, Any], config: FakeConfig) -> dict[str, Any]:
    """Generate fake OpenAI embeddings response."""
    raise NotImplementedError("Fake embeddings not implemented yet")


def generate_fake_models_list(config: FakeConfig) -> dict[str, Any]:
    """Generate fake OpenAI models list response."""
    raise NotImplementedError("Fake models list not implemented yet")


async def generate_fake_stream(
    response_data: Any, endpoint: str, config: FakeConfig
) -> AsyncGenerator[dict[str, Any], None]:
    """Convert fake response to streaming chunks."""
    latency_seconds = config.latency_ms / 1000.0

    if endpoint == "/v1/chat/completions":
        if hasattr(response_data, "choices"):
            content = response_data.choices[0].message.content
            chunk_id = response_data.id
            model = response_data.model
        else:
            content = response_data["choices"][0]["message"]["content"]
            chunk_id = response_data["id"]
            model = response_data["model"]

        words = content.split()

        yield {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "",
                        "function_call": None,
                        "tool_calls": None,
                    },
                    "finish_reason": None,
                    "logprobs": None,
                }
            ],
            "system_fingerprint": None,
        }

        await asyncio.sleep(latency_seconds)

        for i, word in enumerate(words):
            chunk_content = word + (" " if i < len(words) - 1 else "")

            yield {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": chunk_content,
                            "function_call": None,
                            "tool_calls": None,
                        },
                        "finish_reason": None,
                        "logprobs": None,
                    }
                ],
                "system_fingerprint": None,
            }

            await asyncio.sleep(latency_seconds)

        yield {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": None,
                        "function_call": None,
                        "tool_calls": None,
                    },
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "system_fingerprint": None,
        }

    elif endpoint == "/v1/completions":
        raise NotImplementedError("Fake streaming completions not implemented yet")


def generate_fake_response(endpoint: str, body: dict[str, Any], config: FakeConfig) -> Any:
    """Generate fake responses based on endpoint and request."""
    if endpoint == "/v1/chat/completions":
        return generate_fake_chat_completion(body, config)
    elif endpoint == "/v1/completions":
        return generate_fake_completion(body, config)
    elif endpoint == "/v1/embeddings":
        return generate_fake_embeddings(body, config)
    elif endpoint == "/v1/models":
        return generate_fake_models_list(config)
    else:
        raise ValueError(f"Unsupported endpoint for fake mode: {endpoint}")
