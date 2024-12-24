# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import copy
import json
import logging
import os
import re
import secrets
import string
import uuid
from datetime import datetime
from typing import AsyncGenerator, List
from urllib.parse import urlparse

import httpx

from llama_stack.apis.agents import *  # noqa: F403
from llama_stack.apis.common.content_types import TextContentItem
from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.apis.memory_banks import *  # noqa: F403
from llama_stack.apis.safety import *  # noqa: F403
from llama_stack.apis.tools import ToolGroups, ToolRuntime
from llama_stack.providers.utils.kvstore import KVStore
from llama_stack.providers.utils.telemetry import tracing

from .persistence import AgentPersistence
from .safety import SafetyException, ShieldRunnerMixin

log = logging.getLogger(__name__)


def make_random_string(length: int = 8):
    return "".join(
        secrets.choice(string.ascii_letters + string.digits) for _ in range(length)
    )


TOOLS_ATTACHMENT_KEY_REGEX = re.compile(r"__tools_attachment__=(\{.*?\})")


class ChatAgent(ShieldRunnerMixin):
    def __init__(
        self,
        agent_id: str,
        agent_config: AgentConfig,
        tempdir: str,
        inference_api: Inference,
        memory_api: Memory,
        memory_banks_api: MemoryBanks,
        safety_api: Safety,
        tool_runtime_api: ToolRuntime,
        tool_groups_api: ToolGroups,
        persistence_store: KVStore,
    ):
        self.agent_id = agent_id
        self.agent_config = agent_config
        self.tempdir = tempdir
        self.inference_api = inference_api
        self.memory_api = memory_api
        self.memory_banks_api = memory_banks_api
        self.safety_api = safety_api
        self.storage = AgentPersistence(agent_id, persistence_store)
        self.tool_runtime_api = tool_runtime_api
        self.tool_groups_api = tool_groups_api

        ShieldRunnerMixin.__init__(
            self,
            safety_api,
            input_shields=agent_config.input_shields,
            output_shields=agent_config.output_shields,
        )

    def turn_to_messages(self, turn: Turn) -> List[Message]:
        messages = []

        # We do not want to keep adding RAG context to the input messages
        # May be this should be a parameter of the agentic instance
        # that can define its behavior in a custom way
        for m in turn.input_messages:
            msg = m.model_copy()
            if isinstance(msg, UserMessage):
                msg.context = None
            messages.append(msg)

        for step in turn.steps:
            if step.step_type == StepType.inference.value:
                messages.append(step.model_response)
            elif step.step_type == StepType.tool_execution.value:
                for response in step.tool_responses:
                    messages.append(
                        ToolResponseMessage(
                            call_id=response.call_id,
                            tool_name=response.tool_name,
                            content=response.content,
                        )
                    )
            elif step.step_type == StepType.shield_call.value:
                if step.violation:
                    # CompletionMessage itself in the ShieldResponse
                    messages.append(
                        CompletionMessage(
                            content=step.violation.user_message,
                            stop_reason=StopReason.end_of_turn,
                        )
                    )
        return messages

    async def create_session(self, name: str) -> str:
        return await self.storage.create_session(name)

    async def create_and_execute_turn(
        self, request: AgentTurnCreateRequest
    ) -> AsyncGenerator:
        with tracing.span("create_and_execute_turn") as span:
            span.set_attribute("session_id", request.session_id)
            span.set_attribute("agent_id", self.agent_id)
            span.set_attribute("request", request.model_dump_json())
            assert request.stream is True, "Non-streaming not supported"

            session_info = await self.storage.get_session_info(request.session_id)
            if session_info is None:
                raise ValueError(f"Session {request.session_id} not found")

            turns = await self.storage.get_session_turns(request.session_id)

            messages = []
            if self.agent_config.instructions != "":
                messages.append(SystemMessage(content=self.agent_config.instructions))

            for i, turn in enumerate(turns):
                messages.extend(self.turn_to_messages(turn))

            messages.extend(request.messages)

            turn_id = str(uuid.uuid4())
            span.set_attribute("turn_id", turn_id)
            start_time = datetime.now()
            yield AgentTurnResponseStreamChunk(
                event=AgentTurnResponseEvent(
                    payload=AgentTurnResponseTurnStartPayload(
                        turn_id=turn_id,
                    )
                )
            )

            steps = []
            output_message = None
            async for chunk in self.run(
                session_id=request.session_id,
                turn_id=turn_id,
                input_messages=messages,
                attachments=request.attachments or [],
                sampling_params=self.agent_config.sampling_params,
                stream=request.stream,
            ):
                if isinstance(chunk, CompletionMessage):
                    log.info(
                        f"{chunk.role.capitalize()}: {chunk.content}",
                    )
                    output_message = chunk
                    continue

                assert isinstance(
                    chunk, AgentTurnResponseStreamChunk
                ), f"Unexpected type {type(chunk)}"
                event = chunk.event
                if (
                    event.payload.event_type
                    == AgentTurnResponseEventType.step_complete.value
                ):
                    steps.append(event.payload.step_details)

                yield chunk

            assert output_message is not None

            turn = Turn(
                turn_id=turn_id,
                session_id=request.session_id,
                input_messages=request.messages,
                output_message=output_message,
                started_at=start_time,
                completed_at=datetime.now(),
                steps=steps,
            )
            await self.storage.add_turn_to_session(request.session_id, turn)

            chunk = AgentTurnResponseStreamChunk(
                event=AgentTurnResponseEvent(
                    payload=AgentTurnResponseTurnCompletePayload(
                        turn=turn,
                    )
                )
            )
            yield chunk

    async def run(
        self,
        session_id: str,
        turn_id: str,
        input_messages: List[Message],
        attachments: List[Attachment],
        sampling_params: SamplingParams,
        stream: bool = False,
    ) -> AsyncGenerator:
        # Doing async generators makes downstream code much simpler and everything amenable to
        # streaming. However, it also makes things complicated here because AsyncGenerators cannot
        # return a "final value" for the `yield from` statement. we simulate that by yielding a
        # final boolean (to see whether an exception happened) and then explicitly testing for it.

        if len(self.input_shields) > 0:
            async for res in self.run_multiple_shields_wrapper(
                turn_id, input_messages, self.input_shields, "user-input"
            ):
                if isinstance(res, bool):
                    return
                else:
                    yield res

        async for res in self._run(
            session_id, turn_id, input_messages, attachments, sampling_params, stream
        ):
            if isinstance(res, bool):
                return
            elif isinstance(res, CompletionMessage):
                final_response = res
                break
            else:
                yield res

        assert final_response is not None
        # for output shields run on the full input and output combination
        messages = input_messages + [final_response]

        if len(self.output_shields) > 0:
            async for res in self.run_multiple_shields_wrapper(
                turn_id, messages, self.output_shields, "assistant-output"
            ):
                if isinstance(res, bool):
                    return
                else:
                    yield res

        yield final_response

    async def run_multiple_shields_wrapper(
        self,
        turn_id: str,
        messages: List[Message],
        shields: List[str],
        touchpoint: str,
    ) -> AsyncGenerator:
        with tracing.span("run_shields") as span:
            span.set_attribute("input", [m.model_dump_json() for m in messages])
            if len(shields) == 0:
                span.set_attribute("output", "no shields")
                return

            step_id = str(uuid.uuid4())
            try:
                yield AgentTurnResponseStreamChunk(
                    event=AgentTurnResponseEvent(
                        payload=AgentTurnResponseStepStartPayload(
                            step_type=StepType.shield_call.value,
                            step_id=step_id,
                            metadata=dict(touchpoint=touchpoint),
                        )
                    )
                )
                await self.run_multiple_shields(messages, shields)

            except SafetyException as e:
                yield AgentTurnResponseStreamChunk(
                    event=AgentTurnResponseEvent(
                        payload=AgentTurnResponseStepCompletePayload(
                            step_type=StepType.shield_call.value,
                            step_details=ShieldCallStep(
                                step_id=step_id,
                                turn_id=turn_id,
                                violation=e.violation,
                            ),
                        )
                    )
                )
                span.set_attribute("output", e.violation.model_dump_json())

                yield CompletionMessage(
                    content=str(e),
                    stop_reason=StopReason.end_of_turn,
                )
                yield False

            yield AgentTurnResponseStreamChunk(
                event=AgentTurnResponseEvent(
                    payload=AgentTurnResponseStepCompletePayload(
                        step_type=StepType.shield_call.value,
                        step_details=ShieldCallStep(
                            step_id=step_id,
                            turn_id=turn_id,
                            violation=None,
                        ),
                    )
                )
            )
            span.set_attribute("output", "no violations")

    async def _run(
        self,
        session_id: str,
        turn_id: str,
        input_messages: List[Message],
        attachments: List[Attachment],
        sampling_params: SamplingParams,
        stream: bool = False,
    ) -> AsyncGenerator:
        if self.agent_config.preprocessing_tools:
            with tracing.span("preprocessing_tools") as span:
                for tool_name in self.agent_config.preprocessing_tools:
                    yield AgentTurnResponseStreamChunk(
                        event=AgentTurnResponseEvent(
                            payload=AgentTurnResponseStepStartPayload(
                                step_type=StepType.tool_execution.value,
                                step_id=str(uuid.uuid4()),
                            )
                        )
                    )
                    args = dict(
                        session_id=session_id,
                        input_messages=input_messages,
                        attachments=attachments,
                    )
                    result = await self.tool_runtime_api.invoke_tool(
                        tool_name=tool_name,
                        args=args,
                    )
                    yield AgentTurnResponseStreamChunk(
                        event=AgentTurnResponseEvent(
                            payload=AgentTurnResponseStepProgressPayload(
                                step_type=StepType.tool_execution.value,
                                step_id=str(uuid.uuid4()),
                                tool_call_delta=ToolCallDelta(
                                    parse_status=ToolCallParseStatus.success,
                                    content=ToolCall(
                                        call_id="", tool_name=tool_name, arguments={}
                                    ),
                                ),
                            )
                        )
                    )
                    span.set_attribute(
                        "input", [m.model_dump_json() for m in input_messages]
                    )
                    span.set_attribute("output", result.content)
                    span.set_attribute("error_code", result.error_code)
                    span.set_attribute("error_message", result.error_message)
                    span.set_attribute("tool_name", tool_name)
                    if result.error_code != 0 and result.content:
                        last_message = input_messages[-1]
                        last_message.context = result.content

        output_attachments = []

        n_iter = 0
        # Build a map of custom tools to their definitions for faster lookup
        custom_tools = {}
        for tool in self.agent_config.custom_tools:
            custom_tools[tool.name] = tool
        while True:
            msg = input_messages[-1]

            step_id = str(uuid.uuid4())
            yield AgentTurnResponseStreamChunk(
                event=AgentTurnResponseEvent(
                    payload=AgentTurnResponseStepStartPayload(
                        step_type=StepType.inference.value,
                        step_id=step_id,
                    )
                )
            )

            tool_calls = []
            content = ""
            stop_reason = None

            with tracing.span("inference") as span:
                async for chunk in await self.inference_api.chat_completion(
                    self.agent_config.model,
                    input_messages,
                    tools=await self._get_tools(),
                    tool_prompt_format=self.agent_config.tool_prompt_format,
                    stream=True,
                    sampling_params=sampling_params,
                ):
                    event = chunk.event
                    if event.event_type == ChatCompletionResponseEventType.start:
                        continue
                    elif event.event_type == ChatCompletionResponseEventType.complete:
                        stop_reason = StopReason.end_of_turn
                        continue

                    delta = event.delta
                    if isinstance(delta, ToolCallDelta):
                        if delta.parse_status == ToolCallParseStatus.success:
                            tool_calls.append(delta.content)
                        if stream:
                            yield AgentTurnResponseStreamChunk(
                                event=AgentTurnResponseEvent(
                                    payload=AgentTurnResponseStepProgressPayload(
                                        step_type=StepType.inference.value,
                                        step_id=step_id,
                                        text_delta="",
                                        tool_call_delta=delta,
                                    )
                                )
                            )

                    elif isinstance(delta, str):
                        content += delta
                        if stream and event.stop_reason is None:
                            yield AgentTurnResponseStreamChunk(
                                event=AgentTurnResponseEvent(
                                    payload=AgentTurnResponseStepProgressPayload(
                                        step_type=StepType.inference.value,
                                        step_id=step_id,
                                        text_delta=event.delta,
                                    )
                                )
                            )
                    else:
                        raise ValueError(f"Unexpected delta type {type(delta)}")

                    if event.stop_reason is not None:
                        stop_reason = event.stop_reason
                span.set_attribute("stop_reason", stop_reason)
                span.set_attribute(
                    "input", [m.model_dump_json() for m in input_messages]
                )
                span.set_attribute(
                    "output", f"content: {content} tool_calls: {tool_calls}"
                )

            stop_reason = stop_reason or StopReason.out_of_tokens

            # If tool calls are parsed successfully,
            # if content is not made null the tool call str will also be in the content
            # and tokens will have tool call syntax included twice
            if tool_calls:
                content = ""

            message = CompletionMessage(
                content=content,
                stop_reason=stop_reason,
                tool_calls=tool_calls,
            )

            yield AgentTurnResponseStreamChunk(
                event=AgentTurnResponseEvent(
                    payload=AgentTurnResponseStepCompletePayload(
                        step_type=StepType.inference.value,
                        step_id=step_id,
                        step_details=InferenceStep(
                            # somewhere deep, we are re-assigning message or closing over some
                            # variable which causes message to mutate later on. fix with a
                            # `deepcopy` for now, but this is symptomatic of a deeper issue.
                            step_id=step_id,
                            turn_id=turn_id,
                            model_response=copy.deepcopy(message),
                        ),
                    )
                )
            )

            if n_iter >= self.agent_config.max_infer_iters:
                log.info("Done with MAX iterations, exiting.")
                yield message
                break

            if stop_reason == StopReason.out_of_tokens:
                log.info("Out of token budget, exiting.")
                yield message
                break

            if len(message.tool_calls) == 0:
                if stop_reason == StopReason.end_of_turn:
                    # TODO: UPDATE RETURN TYPE TO SEND A TUPLE OF (MESSAGE, ATTACHMENTS)
                    if len(output_attachments) > 0:
                        if isinstance(message.content, list):
                            message.content += attachments
                        else:
                            message.content = [message.content] + attachments
                    yield message
                else:
                    log.info(f"Partial message: {str(message)}")
                    input_messages = input_messages + [message]
            else:
                log.info(f"{str(message)}")
                tool_call = message.tool_calls[0]
                if tool_call.tool_name in custom_tools:
                    yield message
                    return

                step_id = str(uuid.uuid4())
                yield AgentTurnResponseStreamChunk(
                    event=AgentTurnResponseEvent(
                        payload=AgentTurnResponseStepStartPayload(
                            step_type=StepType.tool_execution.value,
                            step_id=step_id,
                        )
                    )
                )
                yield AgentTurnResponseStreamChunk(
                    event=AgentTurnResponseEvent(
                        payload=AgentTurnResponseStepProgressPayload(
                            step_type=StepType.tool_execution.value,
                            step_id=step_id,
                            tool_call=tool_call,
                        )
                    )
                )

                with tracing.span(
                    "tool_execution",
                    {
                        "tool_name": tool_call.tool_name,
                        "input": message.model_dump_json(),
                    },
                ) as span:
                    result_messages = await execute_tool_call_maybe(
                        self.tool_runtime_api,
                        session_id,
                        [message],
                    )
                    assert (
                        len(result_messages) == 1
                    ), "Currently not supporting multiple messages"
                    result_message = result_messages[0]
                    span.set_attribute("output", result_message.model_dump_json())

                yield AgentTurnResponseStreamChunk(
                    event=AgentTurnResponseEvent(
                        payload=AgentTurnResponseStepCompletePayload(
                            step_type=StepType.tool_execution.value,
                            step_details=ToolExecutionStep(
                                step_id=step_id,
                                turn_id=turn_id,
                                tool_calls=[tool_call],
                                tool_responses=[
                                    ToolResponse(
                                        call_id=result_message.call_id,
                                        tool_name=result_message.tool_name,
                                        content=result_message.content,
                                    )
                                ],
                            ),
                        )
                    )
                )

                # TODO: add tool-input touchpoint and a "start" event for this step also
                # but that needs a lot more refactoring of Tool code potentially

                def interpret_content_as_attachment(
                    content: str,
                ) -> Optional[Attachment]:
                    match = re.search(TOOLS_ATTACHMENT_KEY_REGEX, content)
                    if match:
                        snippet = match.group(1)
                        data = json.loads(snippet)
                        return Attachment(
                            url=URL(uri="file://" + data["filepath"]),
                            mime_type=data["mimetype"],
                        )

                    return None

                if out_attachment := interpret_content_as_attachment(
                    result_message.content
                ):
                    # NOTE: when we push this message back to the model, the model may ignore the
                    # attached file path etc. since the model is trained to only provide a user message
                    # with the summary. We keep all generated attachments and then attach them to final message
                    output_attachments.append(out_attachment)

                input_messages = input_messages + [message, result_message]

            n_iter += 1

    async def _get_tools(self) -> List[ToolDefinition]:
        ret = []
        for tool in self.agent_config.custom_tools:
            params = {}
            for param in tool.parameters:
                params[param.name] = ToolParamDefinition(
                    param_type=param.parameter_type,
                    description=param.description,
                    required=param.required,
                    default=param.default,
                )
            ret.append(
                ToolDefinition(
                    tool_name=tool.name,
                    description=tool.description,
                    parameters=params,
                )
            )
        for tool_name in self.agent_config.available_tools:
            tool = await self.tool_groups_api.get_tool(tool_name)
            if tool.built_in_type:
                ret.append(ToolDefinition(tool_name=tool.built_in_type))
                continue
            params = {}
            for param in tool.parameters:
                params[param.name] = ToolParamDefinition(
                    param_type=param.parameter_type,
                    description=param.description,
                    required=param.required,
                    default=param.default,
                )
            ret.append(
                ToolDefinition(
                    tool_name=tool.identifier,
                    description=tool.description,
                    parameters=params,
                )
            )
        return ret


async def attachment_message(tempdir: str, urls: List[URL]) -> ToolResponseMessage:
    content = []

    for url in urls:
        uri = url.uri
        if uri.startswith("file://"):
            filepath = uri[len("file://") :]
        elif uri.startswith("http"):
            path = urlparse(uri).path
            basename = os.path.basename(path)
            filepath = f"{tempdir}/{make_random_string() + basename}"
            log.info(f"Downloading {url} -> {filepath}")

            async with httpx.AsyncClient() as client:
                r = await client.get(uri)
                resp = r.text
                with open(filepath, "w") as fp:
                    fp.write(resp)
        else:
            raise ValueError(f"Unsupported URL {url}")

        content.append(
            TextContentItem(
                text=f'# There is a file accessible to you at "{filepath}"\n'
            )
        )

    return ToolResponseMessage(
        call_id="",
        tool_name=BuiltinTool.code_interpreter,
        content=content,
    )


async def execute_tool_call_maybe(
    tool_runtime_api: ToolRuntime, session_id: str, messages: List[CompletionMessage]
) -> List[ToolResponseMessage]:
    # While Tools.run interface takes a list of messages,
    # All tools currently only run on a single message
    # When this changes, we can drop this assert
    # Whether to call tools on each message and aggregate
    # or aggregate and call tool once, reamins to be seen.
    assert len(messages) == 1, "Expected single message"
    message = messages[0]

    tool_call = message.tool_calls[0]
    name = tool_call.tool_name
    if isinstance(name, BuiltinTool):
        name = name.value
    result = await tool_runtime_api.invoke_tool(
        tool_name=name,
        args=dict(
            session_id=session_id,
            **tool_call.arguments,
        ),
    )
    return [
        ToolResponseMessage(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            content=result.content,
        )
    ]
