# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from jinja2 import Template
from llama_models.llama3.api import *  # noqa: F403


from termcolor import cprint  # noqa: F401

from llama_stack.apis.agents import (
    DefaultMemoryQueryGeneratorConfig,
    LLMMemoryQueryGeneratorConfig,
    MemoryQueryGenerator,
    MemoryQueryGeneratorConfig,
)
from llama_stack.apis.inference import *  # noqa: F403


async def generate_rag_query(
    config: MemoryQueryGeneratorConfig,
    messages: List[Message],
    **kwargs,
):
    """
    Generates a query that will be used for
    retrieving relevant information from the memory bank.
    """
    if config.type == MemoryQueryGenerator.default.value:
        query = await default_rag_query_generator(config, messages, **kwargs)
    elif config.type == MemoryQueryGenerator.llm.value:
        query = await llm_rag_query_generator(config, messages, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported memory query generator {config.type}")
    # cprint(f"Generated query >>>: {query}", color="green")
    return query


async def default_rag_query_generator(
    config: DefaultMemoryQueryGeneratorConfig,
    messages: List[Message],
    **kwargs,
):
    return config.sep.join(interleaved_text_media_as_str(m.content) for m in messages)


async def llm_rag_query_generator(
    config: LLMMemoryQueryGeneratorConfig,
    messages: List[Message],
    **kwargs,
):
    assert "inference_api" in kwargs, "LLMRAGQueryGenerator needs inference_api"
    inference_api = kwargs["inference_api"]

    m_dict = {"messages": [m.model_dump() for m in messages]}

    template = Template(config.template)
    content = template.render(m_dict)

    model = config.model
    message = UserMessage(content=content)
    response = await inference_api.chat_completion(
        model=model,
        messages=[message],
        stream=False,
    )

    query = response.completion_message.content

    return query
