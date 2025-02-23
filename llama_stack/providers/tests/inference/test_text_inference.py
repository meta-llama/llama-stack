# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from llama_stack.apis.common.content_types import ToolCallParseStatus
from llama_stack.apis.inference import (
    ChatCompletionResponse,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    CompletionResponse,
    CompletionResponseStreamChunk,
    JsonSchemaResponseFormat,
    LogProbConfig,
    Message,
    SystemMessage,
    ToolChoice,
    UserMessage,
)
from llama_stack.apis.models import ListModelsResponse, Model
from llama_stack.models.llama.datatypes import (
    SamplingParams,
    StopReason,
    ToolCall,
    ToolPromptFormat,
)
from llama_stack.providers.tests.test_cases.test_case import TestCase

from .utils import group_chunks

# How to run this test:
#
# pytest -v -s llama_stack/providers/tests/inference/test_text_inference.py
#   -m "(fireworks or ollama) and llama_3b"
#   --env FIREWORKS_API_KEY=<your_api_key>


def get_expected_stop_reason(model: str):
    return StopReason.end_of_message if ("Llama3.1" in model or "Llama-3.1" in model) else StopReason.end_of_turn


@pytest.fixture
def common_params(inference_model):
    return {
        "tool_choice": ToolChoice.auto,
        "tool_prompt_format": (
            ToolPromptFormat.json
            if ("Llama3.1" in inference_model or "Llama-3.1" in inference_model)
            else ToolPromptFormat.python_list
        ),
    }


class TestInference:
    # Session scope for asyncio because the tests in this class all
    # share the same provider instance.
    @pytest.mark.asyncio(loop_scope="session")
    async def test_model_list(self, inference_model, inference_stack):
        _, models_impl = inference_stack
        response = await models_impl.list_models()
        assert isinstance(response, ListModelsResponse)
        assert isinstance(response.data, list)
        assert len(response.data) >= 1
        assert all(isinstance(model, Model) for model in response.data)

        model_def = None
        for model in response.data:
            if model.identifier == inference_model:
                model_def = model
                break

        assert model_def is not None

    @pytest.mark.parametrize(
        "test_case",
        [
            "inference:completion:non_streaming",
        ],
    )
    @pytest.mark.asyncio(loop_scope="session")
    async def test_text_completion_non_streaming(self, inference_model, inference_stack, test_case):
        inference_impl, _ = inference_stack

        tc = TestCase(test_case)

        response = await inference_impl.completion(
            content=tc["content"],
            stream=False,
            model_id=inference_model,
            sampling_params=SamplingParams(
                max_tokens=50,
            ),
        )

        assert isinstance(response, CompletionResponse)
        assert tc["expected"] in response.content

    @pytest.mark.parametrize(
        "test_case",
        [
            "inference:completion:streaming",
        ],
    )
    @pytest.mark.asyncio(loop_scope="session")
    async def test_text_completion_streaming(self, inference_model, inference_stack, test_case):
        inference_impl, _ = inference_stack

        tc = TestCase(test_case)

        chunks = [
            r
            async for r in await inference_impl.completion(
                content=tc["content"],
                stream=True,
                model_id=inference_model,
                sampling_params=SamplingParams(
                    max_tokens=50,
                ),
            )
        ]

        assert all(isinstance(chunk, CompletionResponseStreamChunk) for chunk in chunks)
        assert len(chunks) >= 1
        last = chunks[-1]
        assert last.stop_reason == StopReason.out_of_tokens

    @pytest.mark.parametrize(
        "test_case",
        [
            "inference:completion:logprobs_non_streaming",
        ],
    )
    @pytest.mark.asyncio(loop_scope="session")
    async def test_text_completion_logprobs_non_streaming(self, inference_model, inference_stack, test_case):
        inference_impl, _ = inference_stack

        tc = TestCase(test_case)

        response = await inference_impl.completion(
            content=tc["content"],
            stream=False,
            model_id=inference_model,
            sampling_params=SamplingParams(
                max_tokens=5,
            ),
            logprobs=LogProbConfig(
                top_k=3,
            ),
        )

        assert isinstance(response, CompletionResponse)
        assert 1 <= len(response.logprobs) <= 5
        assert response.logprobs, "Logprobs should not be empty"
        assert all(len(logprob.logprobs_by_token) == 3 for logprob in response.logprobs)

    @pytest.mark.parametrize(
        "test_case",
        [
            "inference:completion:logprobs_streaming",
        ],
    )
    @pytest.mark.asyncio(loop_scope="session")
    async def test_text_completion_logprobs_streaming(self, inference_model, inference_stack, test_case):
        inference_impl, _ = inference_stack

        tc = TestCase(test_case)

        chunks = [
            r
            async for r in await inference_impl.completion(
                content=tc["content"],
                stream=True,
                model_id=inference_model,
                sampling_params=SamplingParams(
                    max_tokens=5,
                ),
                logprobs=LogProbConfig(
                    top_k=3,
                ),
            )
        ]

        assert all(isinstance(chunk, CompletionResponseStreamChunk) for chunk in chunks)
        assert (
            1 <= len(chunks) <= 6
        )  # why 6 and not 5? the response may have an extra closing chunk, e.g. for usage or stop_reason
        for chunk in chunks:
            if chunk.delta:  # if there's a token, we expect logprobs
                assert chunk.logprobs, "Logprobs should not be empty"
                assert all(len(logprob.logprobs_by_token) == 3 for logprob in chunk.logprobs)
            else:  # no token, no logprobs
                assert not chunk.logprobs, "Logprobs should be empty"

    @pytest.mark.parametrize(
        "test_case",
        [
            "inference:completion:structured_output",
        ],
    )
    @pytest.mark.asyncio(loop_scope="session")
    async def test_text_completion_structured_output(self, inference_model, inference_stack, test_case):
        inference_impl, _ = inference_stack

        class Output(BaseModel):
            name: str
            year_born: str
            year_retired: str

        tc = TestCase(test_case)

        user_input = tc["user_input"]
        response = await inference_impl.completion(
            model_id=inference_model,
            content=user_input,
            stream=False,
            sampling_params=SamplingParams(
                max_tokens=50,
            ),
            response_format=JsonSchemaResponseFormat(
                json_schema=Output.model_json_schema(),
            ),
        )
        assert isinstance(response, CompletionResponse)
        assert isinstance(response.content, str)

        answer = Output.model_validate_json(response.content)
        expected = tc["expected"]
        assert answer.name == expected["name"]
        assert answer.year_born == expected["year_born"]
        assert answer.year_retired == expected["year_retired"]

    @pytest.mark.parametrize(
        "test_case",
        [
            "inference:chat_completion:sample_messages",
        ],
    )
    @pytest.mark.asyncio(loop_scope="session")
    async def test_text_chat_completion_non_streaming(self, inference_model, inference_stack, common_params, test_case):
        inference_impl, _ = inference_stack
        tc = TestCase(test_case)
        messages = [TypeAdapter(Message).validate_python(m) for m in tc["messages"]]
        response = await inference_impl.chat_completion(
            model_id=inference_model,
            messages=messages,
            stream=False,
            **common_params,
        )

        assert isinstance(response, ChatCompletionResponse)
        assert response.completion_message.role == "assistant"
        assert isinstance(response.completion_message.content, str)
        assert len(response.completion_message.content) > 0

    @pytest.mark.parametrize(
        "test_case",
        [
            "inference:chat_completion:structured_output",
        ],
    )
    @pytest.mark.asyncio(loop_scope="session")
    async def test_text_chat_completion_structured_output(
        self, inference_model, inference_stack, common_params, test_case
    ):
        inference_impl, _ = inference_stack

        class AnswerFormat(BaseModel):
            first_name: str
            last_name: str
            year_of_birth: int
            num_seasons_in_nba: int

        tc = TestCase(test_case)
        messages = [TypeAdapter(Message).validate_python(m) for m in tc["messages"]]

        response = await inference_impl.chat_completion(
            model_id=inference_model,
            messages=messages,
            stream=False,
            response_format=JsonSchemaResponseFormat(
                json_schema=AnswerFormat.model_json_schema(),
            ),
            **common_params,
        )

        assert isinstance(response, ChatCompletionResponse)
        assert response.completion_message.role == "assistant"
        assert isinstance(response.completion_message.content, str)

        answer = AnswerFormat.model_validate_json(response.completion_message.content)
        expected = tc["expected"]
        assert answer.first_name == expected["first_name"]
        assert answer.last_name == expected["last_name"]
        assert answer.year_of_birth == expected["year_of_birth"]
        assert answer.num_seasons_in_nba == expected["num_seasons_in_nba"]

        response = await inference_impl.chat_completion(
            model_id=inference_model,
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content="Please give me information about Michael Jordan."),
            ],
            stream=False,
            **common_params,
        )

        assert isinstance(response, ChatCompletionResponse)
        assert isinstance(response.completion_message.content, str)

        with pytest.raises(ValidationError):
            AnswerFormat.model_validate_json(response.completion_message.content)

    @pytest.mark.parametrize(
        "test_case",
        [
            "inference:chat_completion:sample_messages",
        ],
    )
    @pytest.mark.asyncio(loop_scope="session")
    async def test_text_chat_completion_streaming(self, inference_model, inference_stack, common_params, test_case):
        inference_impl, _ = inference_stack
        tc = TestCase(test_case)
        messages = [TypeAdapter(Message).validate_python(m) for m in tc["messages"]]
        response = [
            r
            async for r in await inference_impl.chat_completion(
                model_id=inference_model,
                messages=messages,
                stream=True,
                **common_params,
            )
        ]

        assert len(response) > 0
        assert all(isinstance(chunk, ChatCompletionResponseStreamChunk) for chunk in response)
        grouped = group_chunks(response)
        assert len(grouped[ChatCompletionResponseEventType.start]) == 1
        assert len(grouped[ChatCompletionResponseEventType.progress]) > 0
        assert len(grouped[ChatCompletionResponseEventType.complete]) == 1

        end = grouped[ChatCompletionResponseEventType.complete][0]
        assert end.event.stop_reason == StopReason.end_of_turn

    @pytest.mark.parametrize(
        "test_case",
        [
            "inference:chat_completion:sample_messages_tool_calling",
        ],
    )
    @pytest.mark.asyncio(loop_scope="session")
    async def test_text_chat_completion_with_tool_calling(
        self,
        inference_model,
        inference_stack,
        common_params,
        test_case,
    ):
        inference_impl, _ = inference_stack
        tc = TestCase(test_case)
        messages = [TypeAdapter(Message).validate_python(m) for m in tc["messages"]]

        response = await inference_impl.chat_completion(
            model_id=inference_model,
            messages=messages,
            tools=tc["tools"],
            stream=False,
            **common_params,
        )

        assert isinstance(response, ChatCompletionResponse)

        message = response.completion_message

        # This is not supported in most providers :/ they don't return eom_id / eot_id
        # stop_reason = get_expected_stop_reason(inference_settings["common_params"]["model"])
        # assert message.stop_reason == stop_reason
        assert message.tool_calls is not None
        assert len(message.tool_calls) > 0

        call = message.tool_calls[0]
        assert call.tool_name == tc["tools"][0]["tool_name"]
        for name, value in tc["expected"].items():
            assert name in call.arguments
            assert value in call.arguments[name]

    @pytest.mark.parametrize(
        "test_case",
        [
            "inference:chat_completion:sample_messages_tool_calling",
        ],
    )
    @pytest.mark.asyncio(loop_scope="session")
    async def test_text_chat_completion_with_tool_calling_streaming(
        self,
        inference_model,
        inference_stack,
        common_params,
        test_case,
    ):
        inference_impl, _ = inference_stack
        tc = TestCase(test_case)
        messages = [TypeAdapter(Message).validate_python(m) for m in tc["messages"]]

        response = [
            r
            async for r in await inference_impl.chat_completion(
                model_id=inference_model,
                messages=messages,
                tools=tc["tools"],
                stream=True,
                **common_params,
            )
        ]
        assert len(response) > 0
        assert all(isinstance(chunk, ChatCompletionResponseStreamChunk) for chunk in response)
        grouped = group_chunks(response)
        assert len(grouped[ChatCompletionResponseEventType.start]) == 1
        assert len(grouped[ChatCompletionResponseEventType.progress]) > 0
        assert len(grouped[ChatCompletionResponseEventType.complete]) == 1

        # This is not supported in most providers :/ they don't return eom_id / eot_id
        # expected_stop_reason = get_expected_stop_reason(
        #     inference_settings["common_params"]["model"]
        # )
        # end = grouped[ChatCompletionResponseEventType.complete][0]
        # assert end.event.stop_reason == expected_stop_reason

        if "Llama3.1" in inference_model:
            assert all(
                chunk.event.delta.type == "tool_call" for chunk in grouped[ChatCompletionResponseEventType.progress]
            )
            first = grouped[ChatCompletionResponseEventType.progress][0]
            if not isinstance(first.event.delta.tool_call, ToolCall):  # first chunk may contain entire call
                assert first.event.delta.parse_status == ToolCallParseStatus.started

        last = grouped[ChatCompletionResponseEventType.progress][-1]
        # assert last.event.stop_reason == expected_stop_reason
        assert last.event.delta.parse_status == ToolCallParseStatus.succeeded
        assert isinstance(last.event.delta.tool_call, ToolCall)

        call = last.event.delta.tool_call
        assert call.tool_name == tc["tools"][0]["tool_name"]
        for name, value in tc["expected"].items():
            assert name in call.arguments
            assert value in call.arguments[name]
