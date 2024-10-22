# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import warnings
from typing import Any, Dict, List, Optional, Tuple

import httpx
from llama_models.llama3.api.datatypes import (
    CompletionMessage,
    StopReason,
    TokenLogProbs,
    ToolCall,
)

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
)

from ._config import NVIDIAConfig


def convert_message(message: Message) -> dict:
    """
    Convert a Message to an OpenAI API-compatible dictionary.
    """
    out_dict = message.dict()
    # Llama Stack uses role="ipython" for tool call messages, OpenAI uses "tool"
    if out_dict["role"] == "ipython":
        out_dict.update(role="tool")

    if "stop_reason" in out_dict:
        out_dict.update(stop_reason=out_dict["stop_reason"].value)

    # TODO(mf): tool_calls

    return out_dict


async def _get_health(url: str) -> Tuple[bool, bool]:
    """
    Query {url}/v1/health/{live,ready} to check if the server is running and ready

    Args:
        url (str): URL of the server

    Returns:
        Tuple[bool, bool]: (is_live, is_ready)
    """
    async with httpx.AsyncClient() as client:
        live = await client.get(f"{url}/v1/health/live")
        ready = await client.get(f"{url}/v1/health/ready")
        return live.status_code == 200, ready.status_code == 200


async def check_health(config: NVIDIAConfig) -> None:
    """
    Check if the server is running and ready

    Args:
        url (str): URL of the server

    Raises:
        RuntimeError: If the server is not running or ready
    """
    if not config.is_hosted:
        print("Checking NVIDIA NIM health...")
        try:
            is_live, is_ready = await _get_health(config.base_url)
            if not is_live:
                raise ConnectionError("NVIDIA NIM is not running")
            if not is_ready:
                raise ConnectionError("NVIDIA NIM is not ready")
            # TODO(mf): should we wait for the server to be ready?
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect to NVIDIA NIM: {e}") from e


def convert_chat_completion_request(
    request: ChatCompletionRequest,
    n: int = 1,
) -> dict:
    """
    Convert a ChatCompletionRequest to an OpenAI API-compatible dictionary.
    """
    # model -> model
    # messages -> messages
    # sampling_params  TODO(mattf): review strategy
    #  strategy=greedy -> nvext.top_k = -1, temperature = temperature
    #  strategy=top_p -> nvext.top_k = -1, top_p = top_p
    #  strategy=top_k -> nvext.top_k = top_k
    #  temperature -> temperature
    #  top_p -> top_p
    #  top_k -> nvext.top_k
    #  max_tokens -> max_tokens
    #  repetition_penalty -> nvext.repetition_penalty
    # tools -> tools
    # tool_choice ("auto", "required") -> tool_choice
    # tool_prompt_format -> TBD
    # stream -> stream
    # logprobs -> logprobs

    print(f"sampling_params: {request.sampling_params}")

    payload: Dict[str, Any] = dict(
        model=request.model,
        messages=[convert_message(message) for message in request.messages],
        stream=request.stream,
        nvext={},
        n=n,
    )
    nvext = payload["nvext"]

    if request.tools:
        payload.update(tools=request.tools)
        if request.tool_choice:
            payload.update(
                tool_choice=request.tool_choice.value
            )  # we cannot include tool_choice w/o tools, server will complain

    if request.logprobs:
        payload.update(logprobs=True)
        payload.update(top_logprobs=request.logprobs.top_k)

    if request.sampling_params:
        nvext.update(repetition_penalty=request.sampling_params.repetition_penalty)

        if request.sampling_params.max_tokens:
            payload.update(max_tokens=request.sampling_params.max_tokens)

        if request.sampling_params.strategy == "top_p":
            nvext.update(top_k=-1)
            payload.update(top_p=request.sampling_params.top_p)
        elif request.sampling_params.strategy == "top_k":
            if (
                request.sampling_params.top_k != -1
                and request.sampling_params.top_k < 1
            ):
                warnings.warn("top_k must be -1 or >= 1")
            nvext.update(top_k=request.sampling_params.top_k)
        elif request.sampling_params.strategy == "greedy":
            nvext.update(top_k=-1)
            payload.update(temperature=request.sampling_params.temperature)

    return payload


def _parse_content(completion: dict) -> str:
    """
    Get the content from an OpenAI completion response.

    OpenAI completion response format -
        {
            ...
            "message": {"role": "assistant", "content": ..., ...},
            ...
        }
    """
    # content is nullable in the OpenAI response, common for tool calls
    return completion["message"]["content"] or ""


def _parse_stop_reason(completion: dict) -> StopReason:
    """
    Get the StopReason from an OpenAI completion response.

    OpenAI completion response format -
        {
            ...
            "finish_reason": "length" or "stop" or "tool_calls",
            ...
        }
    """

    # StopReason options are end_of_turn, end_of_message, out_of_tokens
    # TODO(mf): is end_of_turn and end_of_message usage correct?
    stop_reason = StopReason.end_of_turn
    if completion["finish_reason"] == "length":
        stop_reason = StopReason.out_of_tokens
    elif completion["finish_reason"] == "stop":
        stop_reason = StopReason.end_of_message
    elif completion["finish_reason"] == "tool_calls":
        stop_reason = StopReason.end_of_turn
    return stop_reason


def _parse_tool_calls(completion: dict) -> List[ToolCall]:
    """
    Get the tool calls from an OpenAI completion response.

    OpenAI completion response format -
        {
            ...,
            "message": {
                ...,
                "tool_calls": [
                    {
                        "id": X,
                        "type": "function",
                        "function": {
                            "name": Y,
                            "arguments": Z,
                        },
                    }*
                ],
            },
        }
    ->
        [
            ToolCall(call_id=X, tool_name=Y, arguments=Z),
            ...
        ]
    """
    tool_calls = []
    if "tool_calls" in completion["message"]:
        assert isinstance(
            completion["message"]["tool_calls"], list
        ), "error in server response: tool_calls not a list"
        for call in completion["message"]["tool_calls"]:
            assert "id" in call, "error in server response: tool call id not found"
            assert (
                "function" in call
            ), "error in server response: tool call function not found"
            assert (
                "name" in call["function"]
            ), "error in server response: tool call function name not found"
            assert (
                "arguments" in call["function"]
            ), "error in server response: tool call function arguments not found"
            tool_calls.append(
                ToolCall(
                    call_id=call["id"],
                    tool_name=call["function"]["name"],
                    arguments=call["function"]["arguments"],
                )
            )

    return tool_calls


def _parse_logprobs(completion: dict) -> Optional[List[TokenLogProbs]]:
    """
    Extract logprobs from OpenAI as a list of TokenLogProbs.

    OpenAI completion response format -
        {
            ...
            "logprobs": {
                content: [
                    {
                        ...,
                        top_logprobs: [{token: X, logprob: Y, bytes: [...]}+]
                    }+
                ]
            },
            ...
        }
    ->
        [
            TokenLogProbs(
                logprobs_by_token={X: Y, ...}
            ),
            ...
        ]
    """
    if not (logprobs := completion.get("logprobs")):
        return None

    return [
        TokenLogProbs(
            logprobs_by_token={
                logprobs["token"]: logprobs["logprob"]
                for logprobs in content["top_logprobs"]
            }
        )
        for content in logprobs["content"]
    ]


def parse_completion(
    completion: dict,
) -> ChatCompletionResponse:
    """
    Parse an OpenAI completion response into a CompletionMessage and logprobs.

    OpenAI completion response format -
        {
            "message": {
                "role": "assistant",
                "content": ...,
                "tool_calls": [
                    {
                        ...
                        "id": ...,
                        "function": {
                            "name": ...,
                            "arguments": ...,
                        },
                    }*
                ]?,
            "finish_reason": ...,
            "logprobs": {
                "content": [
                    {
                        ...,
                        "top_logprobs": [{"token": ..., "logprob": ..., ...}+]
                    }+
                ]
            }?
        }
    """
    assert "message" in completion, "error in server response: message not found"
    assert (
        "finish_reason" in completion
    ), "error in server response: finish_reason not found"

    return ChatCompletionResponse(
        completion_message=CompletionMessage(
            content=_parse_content(completion),
            stop_reason=_parse_stop_reason(completion),
            tool_calls=_parse_tool_calls(completion),
        ),
        logprobs=_parse_logprobs(completion),
    )
