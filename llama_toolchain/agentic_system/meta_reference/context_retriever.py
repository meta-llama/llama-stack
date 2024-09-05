# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from jinja2 import Template
from llama_models.llama3.api import *  # noqa: F403


from termcolor import cprint

from llama_toolchain.agentic_system.api import (
    DefaultMemoryQueryGeneratorConfig,
    LLMMemoryQueryGeneratorConfig,
    MemoryQueryGenerator,
    MemoryQueryGeneratorConfig,
)
from llama_toolchain.inference.api import *  # noqa: F403
from llama_toolchain.inference.client import InferenceClient


async def generate_rag_query(
    generator_config: MemoryQueryGeneratorConfig,
    messages: List[Message],
):
    if generator_config.type == MemoryQueryGenerator.default.value:
        generator = DefaultRAGQueryGenerator(generator_config)
    elif generator_config.type == MemoryQueryGenerator.llm.value:
        generator = LLMRAGQueryGenerator(generator_config)
    else:
        raise NotImplementedError(
            f"Unsupported memory query generator {generator_config.type}"
        )

    query = await generator.gen(messages)
    cprint(f"Generated query >>>: {query}", color="green")
    return query


class DefaultRAGQueryGenerator:
    def __init__(self, config: DefaultMemoryQueryGeneratorConfig):
        self.config = config

    async def gen(self, messages: List[Message]) -> InterleavedTextMedia:
        query = self.config.sep.join(
            interleaved_text_media_as_str(m.content) for m in messages
        )
        return query


class LLMRAGQueryGenerator:
    def __init__(self, config: LLMMemoryQueryGeneratorConfig):
        self.config = config

    async def gen(self, messages: List[Message]) -> InterleavedTextMedia:
        # params will have
        """
        Generates a query that will be used for
        retrieving relevant information from the memory bank.
        """
        # get template from user
        # user template will assume data has the format of
        # pydantic object representing List[Message]
        m_dict = {"messages": [m.model_dump() for m in messages]}

        template = Template(self.config.template)
        content = template.render(m_dict)

        cprint(f"Rendered Template >>>: {content}", color="yellow")
        # TODO: How to manage these config params better ?
        host = self.config.host
        port = self.config.port
        client = InferenceClient(f"http://{host}:{port}")

        model = self.config.model
        message = UserMessage(content=content)
        response = client.chat_completion(
            ChatCompletionRequest(
                model=model,
                messages=[message],
                stream=False,
            )
        )

        async for chunk in response:
            query = chunk.completion_message.content

        return query
