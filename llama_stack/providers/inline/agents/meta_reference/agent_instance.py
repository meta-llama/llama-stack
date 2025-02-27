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
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from llama_stack.apis.agents import (
    AgentConfig,
    AgentToolGroup,
    AgentToolGroupWithArgs,
    AgentTurnCreateRequest,
    AgentTurnResponseEvent,
    AgentTurnResponseEventType,
    AgentTurnResponseStepCompletePayload,
    AgentTurnResponseStepProgressPayload,
    AgentTurnResponseStepStartPayload,
    AgentTurnResponseStreamChunk,
    AgentTurnResponseTurnAwaitingInputPayload,
    AgentTurnResponseTurnCompletePayload,
    AgentTurnResponseTurnStartPayload,
    AgentTurnResumeRequest,
    Attachment,
    Document,
    InferenceStep,
    ShieldCallStep,
    StepType,
    ToolExecutionStep,
    Turn,
)
from llama_stack.apis.common.content_types import (
    URL,
    TextContentItem,
    ToolCallDelta,
    ToolCallParseStatus,
)
from llama_stack.apis.inference import (
    ChatCompletionResponseEventType,
    CompletionMessage,
    Inference,
    Message,
    SamplingParams,
    StopReason,
    SystemMessage,
    ToolDefinition,
    ToolResponse,
    ToolResponseMessage,
    UserMessage,
)
from llama_stack.apis.safety import Safety
from llama_stack.apis.tools import (
    RAGDocument,
    ToolGroups,
    ToolInvocationResult,
    ToolRuntime,
)
from llama_stack.apis.vector_io import VectorIO
from llama_stack.models.llama.datatypes import (
    BuiltinTool,
    ToolCall,
    ToolParamDefinition,
)
from llama_stack.providers.utils.kvstore import KVStore
from llama_stack.providers.utils.telemetry import tracing

from .persistence import AgentPersistence
from .safety import SafetyException, ShieldRunnerMixin

log = logging.getLogger(__name__)


def make_random_string(length: int = 8):
    return "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(length))


TOOLS_ATTACHMENT_KEY_REGEX = re.compile(r"__tools_attachment__=(\{.*?\})")
MEMORY_QUERY_TOOL = "knowledge_search"
WEB_SEARCH_TOOL = "web_search"
RAG_TOOL_GROUP = "builtin::rag"


class ChatAgent(ShieldRunnerMixin):
    def __init__(
        self,
        agent_id: str,
        agent_config: AgentConfig,
        tempdir: str,
        inference_api: Inference,
        safety_api: Safety,
        tool_runtime_api: ToolRuntime,
        tool_groups_api: ToolGroups,
        vector_io_api: VectorIO,
        persistence_store: KVStore,
    ):
        self.agent_id = agent_id
        self.agent_config = agent_config
        self.tempdir = tempdir
        self.inference_api = inference_api
        self.safety_api = safety_api
        self.vector_io_api = vector_io_api
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

        for m in turn.input_messages:
            msg = m.model_copy()
            # We do not want to keep adding RAG context to the input messages
            # May be this should be a parameter of the agentic instance
            # that can define its behavior in a custom way
            if isinstance(msg, UserMessage):
                msg.context = None
            if isinstance(msg, ToolResponseMessage):
                # NOTE: do not add ToolResponseMessage here, we'll add them in tool_execution steps
                continue

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

    async def get_messages_from_turns(self, turns: List[Turn]) -> List[Message]:
        messages = []
        if self.agent_config.instructions != "":
            messages.append(SystemMessage(content=self.agent_config.instructions))

        for turn in turns:
            messages.extend(self.turn_to_messages(turn))
        return messages

    async def create_and_execute_turn(self, request: AgentTurnCreateRequest) -> AsyncGenerator:
        with tracing.span("create_and_execute_turn") as span:
            span.set_attribute("session_id", request.session_id)
            span.set_attribute("agent_id", self.agent_id)
            span.set_attribute("request", request.model_dump_json())
            assert request.stream is True, "Non-streaming not supported"

            session_info = await self.storage.get_session_info(request.session_id)
            if session_info is None:
                raise ValueError(f"Session {request.session_id} not found")

            turns = await self.storage.get_session_turns(request.session_id)
            messages = await self.get_messages_from_turns(turns)
            messages.extend(request.messages)

            turn_id = str(uuid.uuid4())
            span.set_attribute("turn_id", turn_id)
            start_time = datetime.now().astimezone().isoformat()
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
                sampling_params=self.agent_config.sampling_params,
                stream=request.stream,
                documents=request.documents,
                toolgroups_for_turn=request.toolgroups,
            ):
                if isinstance(chunk, CompletionMessage):
                    log.info(
                        f"{chunk.role.capitalize()}: {chunk.content}",
                    )
                    output_message = chunk
                    continue

                assert isinstance(chunk, AgentTurnResponseStreamChunk), f"Unexpected type {type(chunk)}"
                event = chunk.event
                if event.payload.event_type == AgentTurnResponseEventType.step_complete.value:
                    steps.append(event.payload.step_details)

                yield chunk

            assert output_message is not None

            turn = Turn(
                turn_id=turn_id,
                session_id=request.session_id,
                input_messages=request.messages,
                output_message=output_message,
                started_at=start_time,
                completed_at=datetime.now().astimezone().isoformat(),
                steps=steps,
            )
            await self.storage.add_turn_to_session(request.session_id, turn)

            if output_message.tool_calls and request.allow_turn_resume:
                chunk = AgentTurnResponseStreamChunk(
                    event=AgentTurnResponseEvent(
                        payload=AgentTurnResponseTurnAwaitingInputPayload(
                            turn=turn,
                        )
                    )
                )
            else:
                chunk = AgentTurnResponseStreamChunk(
                    event=AgentTurnResponseEvent(
                        payload=AgentTurnResponseTurnCompletePayload(
                            turn=turn,
                        )
                    )
                )

            yield chunk

    async def resume_turn(self, request: AgentTurnResumeRequest) -> AsyncGenerator:
        with tracing.span("resume_turn") as span:
            span.set_attribute("agent_id", self.agent_id)
            span.set_attribute("session_id", request.session_id)
            span.set_attribute("turn_id", request.turn_id)
            span.set_attribute("request", request.model_dump_json())
            assert request.stream is True, "Non-streaming not supported"

            session_info = await self.storage.get_session_info(request.session_id)
            if session_info is None:
                raise ValueError(f"Session {request.session_id} not found")

            turns = await self.storage.get_session_turns(request.session_id)
            if len(turns) == 0:
                raise ValueError("No turns found for session")

            messages = await self.get_messages_from_turns(turns)
            messages.extend(request.tool_responses)

            last_turn = turns[-1]
            last_turn_messages = self.turn_to_messages(last_turn)
            last_turn_messages = [
                x for x in last_turn_messages if isinstance(x, UserMessage) or isinstance(x, ToolResponseMessage)
            ]

            # TODO: figure out whether we should add the tool responses to the last turn messages
            last_turn_messages.extend(request.tool_responses)

            # get the steps from the turn id
            steps = []
            steps = turns[-1].steps

            # mark tool execution step as complete
            # if there's no tool execution in progress step (due to storage, or tool call parsing on client),
            # we'll create a new tool execution step with current time
            in_progress_tool_call_step = await self.storage.get_in_progress_tool_call_step(
                request.session_id, request.turn_id
            )
            now = datetime.now().astimezone().isoformat()
            tool_execution_step = ToolExecutionStep(
                step_id=(in_progress_tool_call_step.step_id if in_progress_tool_call_step else str(uuid.uuid4())),
                turn_id=request.turn_id,
                tool_calls=(in_progress_tool_call_step.tool_calls if in_progress_tool_call_step else []),
                tool_responses=[
                    ToolResponse(
                        call_id=x.call_id,
                        tool_name=x.tool_name,
                        content=x.content,
                    )
                    for x in request.tool_responses
                ],
                completed_at=now,
                started_at=(in_progress_tool_call_step.started_at if in_progress_tool_call_step else now),
            )
            steps.append(tool_execution_step)
            yield AgentTurnResponseStreamChunk(
                event=AgentTurnResponseEvent(
                    payload=AgentTurnResponseStepCompletePayload(
                        step_type=StepType.tool_execution.value,
                        step_id=tool_execution_step.step_id,
                        step_details=tool_execution_step,
                    )
                )
            )

            output_message = None
            async for chunk in self.run(
                session_id=request.session_id,
                turn_id=request.turn_id,
                input_messages=messages,
                sampling_params=self.agent_config.sampling_params,
                stream=request.stream,
            ):
                if isinstance(chunk, CompletionMessage):
                    output_message = chunk
                    continue

                assert isinstance(chunk, AgentTurnResponseStreamChunk), f"Unexpected type {type(chunk)}"
                event = chunk.event
                if event.payload.event_type == AgentTurnResponseEventType.step_complete.value:
                    steps.append(event.payload.step_details)

                yield chunk

            assert output_message is not None

            last_turn_start_time = datetime.now().astimezone().isoformat()
            if len(turns) > 0:
                last_turn_start_time = turns[-1].started_at

            turn = Turn(
                turn_id=request.turn_id,
                session_id=request.session_id,
                input_messages=last_turn_messages,
                output_message=output_message,
                started_at=last_turn_start_time,
                completed_at=datetime.now().astimezone().isoformat(),
                steps=steps,
            )
            await self.storage.add_turn_to_session(request.session_id, turn)

            if output_message.tool_calls:
                chunk = AgentTurnResponseStreamChunk(
                    event=AgentTurnResponseEvent(
                        payload=AgentTurnResponseTurnAwaitingInputPayload(
                            turn=turn,
                        )
                    )
                )
            else:
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
        sampling_params: SamplingParams,
        stream: bool = False,
        documents: Optional[List[Document]] = None,
        toolgroups_for_turn: Optional[List[AgentToolGroup]] = None,
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
            session_id,
            turn_id,
            input_messages,
            sampling_params,
            stream,
            documents,
            toolgroups_for_turn,
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
            shield_call_start_time = datetime.now().astimezone().isoformat()
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
                            step_id=step_id,
                            step_details=ShieldCallStep(
                                step_id=step_id,
                                turn_id=turn_id,
                                violation=e.violation,
                                started_at=shield_call_start_time,
                                completed_at=datetime.now().astimezone().isoformat(),
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
                        step_id=step_id,
                        step_details=ShieldCallStep(
                            step_id=step_id,
                            turn_id=turn_id,
                            violation=None,
                            started_at=shield_call_start_time,
                            completed_at=datetime.now().astimezone().isoformat(),
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
        sampling_params: SamplingParams,
        stream: bool = False,
        documents: Optional[List[Document]] = None,
        toolgroups_for_turn: Optional[List[AgentToolGroup]] = None,
    ) -> AsyncGenerator:
        # TODO: simplify all of this code, it can be simpler
        toolgroup_args = {}
        toolgroups = set()
        for toolgroup in self.agent_config.toolgroups + (toolgroups_for_turn or []):
            if isinstance(toolgroup, AgentToolGroupWithArgs):
                tool_group_name, tool_name = self._parse_toolgroup_name(toolgroup.name)
                toolgroups.add(tool_group_name)
                toolgroup_args[tool_group_name] = toolgroup.args
            else:
                toolgroups.add(toolgroup)

        tool_defs, tool_to_group = await self._get_tool_defs(toolgroups_for_turn)
        if documents:
            await self.handle_documents(session_id, documents, input_messages, tool_defs)

        output_attachments = []

        n_iter = 0
        # Build a map of custom tools to their definitions for faster lookup
        client_tools = {}
        for tool in self.agent_config.client_tools:
            client_tools[tool.name] = tool
        while True:
            step_id = str(uuid.uuid4())
            inference_start_time = datetime.now().astimezone().isoformat()
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
                    tools=tool_defs,
                    tool_prompt_format=self.agent_config.tool_config.tool_prompt_format,
                    response_format=self.agent_config.response_format,
                    stream=True,
                    sampling_params=sampling_params,
                    tool_config=self.agent_config.tool_config,
                ):
                    event = chunk.event
                    if event.event_type == ChatCompletionResponseEventType.start:
                        continue
                    elif event.event_type == ChatCompletionResponseEventType.complete:
                        stop_reason = StopReason.end_of_turn
                        continue

                    delta = event.delta
                    if delta.type == "tool_call":
                        if delta.parse_status == ToolCallParseStatus.succeeded:
                            tool_calls.append(delta.tool_call)
                        elif delta.parse_status == ToolCallParseStatus.failed:
                            # If we cannot parse the tools, set the content to the unparsed raw text
                            content = delta.tool_call
                        if stream:
                            yield AgentTurnResponseStreamChunk(
                                event=AgentTurnResponseEvent(
                                    payload=AgentTurnResponseStepProgressPayload(
                                        step_type=StepType.inference.value,
                                        step_id=step_id,
                                        delta=delta,
                                    )
                                )
                            )

                    elif delta.type == "text":
                        content += delta.text
                        if stream and event.stop_reason is None:
                            yield AgentTurnResponseStreamChunk(
                                event=AgentTurnResponseEvent(
                                    payload=AgentTurnResponseStepProgressPayload(
                                        step_type=StepType.inference.value,
                                        step_id=step_id,
                                        delta=delta,
                                    )
                                )
                            )
                    else:
                        raise ValueError(f"Unexpected delta type {type(delta)}")

                    if event.stop_reason is not None:
                        stop_reason = event.stop_reason
                span.set_attribute("stop_reason", stop_reason)
                span.set_attribute("input", [m.model_dump_json() for m in input_messages])
                span.set_attribute("output", f"content: {content} tool_calls: {tool_calls}")

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
                            started_at=inference_start_time,
                            completed_at=datetime.now().astimezone().isoformat(),
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
                            message.content += output_attachments
                        else:
                            message.content = [message.content] + output_attachments
                    yield message
                else:
                    log.info(f"Partial message: {str(message)}")
                    input_messages = input_messages + [message]
            else:
                log.info(f"{str(message)}")
                # 1. Start the tool execution step and progress
                step_id = str(uuid.uuid4())
                yield AgentTurnResponseStreamChunk(
                    event=AgentTurnResponseEvent(
                        payload=AgentTurnResponseStepStartPayload(
                            step_type=StepType.tool_execution.value,
                            step_id=step_id,
                        )
                    )
                )
                tool_call = message.tool_calls[0]
                yield AgentTurnResponseStreamChunk(
                    event=AgentTurnResponseEvent(
                        payload=AgentTurnResponseStepProgressPayload(
                            step_type=StepType.tool_execution.value,
                            step_id=step_id,
                            tool_call=tool_call,
                            delta=ToolCallDelta(
                                parse_status=ToolCallParseStatus.in_progress,
                                tool_call=tool_call,
                            ),
                        )
                    )
                )

                # If tool is a client tool, yield CompletionMessage and return
                if tool_call.tool_name in client_tools:
                    await self.storage.set_in_progress_tool_call_step(
                        session_id,
                        turn_id,
                        ToolExecutionStep(
                            step_id=step_id,
                            turn_id=turn_id,
                            tool_calls=[tool_call],
                            tool_responses=[],
                            started_at=datetime.now().astimezone().isoformat(),
                        ),
                    )
                    yield message
                    return

                # If tool is a builtin server tool, execute it
                tool_name = tool_call.tool_name
                if isinstance(tool_name, BuiltinTool):
                    tool_name = tool_name.value
                with tracing.span(
                    "tool_execution",
                    {
                        "tool_name": tool_name,
                        "input": message.model_dump_json(),
                    },
                ) as span:
                    tool_execution_start_time = datetime.now().astimezone().isoformat()
                    tool_call = message.tool_calls[0]
                    tool_result = await execute_tool_call_maybe(
                        self.tool_runtime_api,
                        session_id,
                        tool_call,
                        toolgroup_args,
                        tool_to_group,
                    )
                    if tool_result.content is None:
                        raise ValueError(
                            f"Tool call result (id: {tool_call.call_id}, name: {tool_call.tool_name}) does not have any content"
                        )
                    result_messages = [
                        ToolResponseMessage(
                            call_id=tool_call.call_id,
                            tool_name=tool_call.tool_name,
                            content=tool_result.content,
                        )
                    ]
                    assert len(result_messages) == 1, "Currently not supporting multiple messages"
                    result_message = result_messages[0]
                    span.set_attribute("output", result_message.model_dump_json())

                yield AgentTurnResponseStreamChunk(
                    event=AgentTurnResponseEvent(
                        payload=AgentTurnResponseStepCompletePayload(
                            step_type=StepType.tool_execution.value,
                            step_id=step_id,
                            step_details=ToolExecutionStep(
                                step_id=step_id,
                                turn_id=turn_id,
                                tool_calls=[tool_call],
                                tool_responses=[
                                    ToolResponse(
                                        call_id=result_message.call_id,
                                        tool_name=result_message.tool_name,
                                        content=result_message.content,
                                        metadata=tool_result.metadata,
                                    )
                                ],
                                started_at=tool_execution_start_time,
                                completed_at=datetime.now().astimezone().isoformat(),
                            ),
                        )
                    )
                )

                # TODO: add tool-input touchpoint and a "start" event for this step also
                # but that needs a lot more refactoring of Tool code potentially
                if (type(result_message.content) is str) and (
                    out_attachment := _interpret_content_as_attachment(result_message.content)
                ):
                    # NOTE: when we push this message back to the model, the model may ignore the
                    # attached file path etc. since the model is trained to only provide a user message
                    # with the summary. We keep all generated attachments and then attach them to final message
                    output_attachments.append(out_attachment)

                input_messages = input_messages + [message, result_message]

            n_iter += 1

    async def _get_tool_defs(
        self, toolgroups_for_turn: Optional[List[AgentToolGroup]] = None
    ) -> Tuple[List[ToolDefinition], Dict[str, str]]:
        # Determine which tools to include
        agent_config_toolgroups = set(
            (toolgroup.name if isinstance(toolgroup, AgentToolGroupWithArgs) else toolgroup)
            for toolgroup in self.agent_config.toolgroups
        )
        toolgroups_for_turn_set = (
            agent_config_toolgroups
            if toolgroups_for_turn is None
            else {
                (toolgroup.name if isinstance(toolgroup, AgentToolGroupWithArgs) else toolgroup)
                for toolgroup in toolgroups_for_turn
            }
        )

        tool_name_to_def = {}
        tool_to_group = {}

        for tool_def in self.agent_config.client_tools:
            if tool_name_to_def.get(tool_def.name, None):
                raise ValueError(f"Tool {tool_def.name} already exists")
            tool_name_to_def[tool_def.name] = ToolDefinition(
                tool_name=tool_def.name,
                description=tool_def.description,
                parameters={
                    param.name: ToolParamDefinition(
                        param_type=param.parameter_type,
                        description=param.description,
                        required=param.required,
                        default=param.default,
                    )
                    for param in tool_def.parameters
                },
            )
            tool_to_group[tool_def.name] = "__client_tools__"
        for toolgroup_name_with_maybe_tool_name in agent_config_toolgroups:
            if toolgroup_name_with_maybe_tool_name not in toolgroups_for_turn_set:
                continue

            toolgroup_name, tool_name = self._parse_toolgroup_name(toolgroup_name_with_maybe_tool_name)
            tools = await self.tool_groups_api.list_tools(toolgroup_id=toolgroup_name)
            if not tools.data:
                available_tool_groups = ", ".join(
                    [t.identifier for t in (await self.tool_groups_api.list_tool_groups()).data]
                )
                raise ValueError(f"Toolgroup {toolgroup_name} not found, available toolgroups: {available_tool_groups}")
            if tool_name is not None and not any(tool.identifier == tool_name for tool in tools.data):
                raise ValueError(
                    f"Tool {tool_name} not found in toolgroup {toolgroup_name}. Available tools: {', '.join([tool.identifier for tool in tools.data])}"
                )

            for tool_def in tools.data:
                if toolgroup_name.startswith("builtin") and toolgroup_name != RAG_TOOL_GROUP:
                    tool_name = tool_def.identifier
                    built_in_type = BuiltinTool.brave_search
                    if tool_name == "web_search":
                        built_in_type = BuiltinTool.brave_search
                    else:
                        built_in_type = BuiltinTool(tool_name)

                    if tool_name_to_def.get(built_in_type, None):
                        raise ValueError(f"Tool {built_in_type} already exists")

                    tool_name_to_def[built_in_type] = ToolDefinition(
                        tool_name=built_in_type,
                        description=tool_def.description,
                        parameters={
                            param.name: ToolParamDefinition(
                                param_type=param.parameter_type,
                                description=param.description,
                                required=param.required,
                                default=param.default,
                            )
                            for param in tool_def.parameters
                        },
                    )
                    tool_to_group[built_in_type] = tool_def.toolgroup_id
                    continue

                if tool_name_to_def.get(tool_def.identifier, None):
                    raise ValueError(f"Tool {tool_def.identifier} already exists")
                if tool_name in (None, tool_def.identifier):
                    tool_name_to_def[tool_def.identifier] = ToolDefinition(
                        tool_name=tool_def.identifier,
                        description=tool_def.description,
                        parameters={
                            param.name: ToolParamDefinition(
                                param_type=param.parameter_type,
                                description=param.description,
                                required=param.required,
                                default=param.default,
                            )
                            for param in tool_def.parameters
                        },
                    )
                    tool_to_group[tool_def.identifier] = tool_def.toolgroup_id

        return list(tool_name_to_def.values()), tool_to_group

    def _parse_toolgroup_name(self, toolgroup_name_with_maybe_tool_name: str) -> tuple[str, Optional[str]]:
        """Parse a toolgroup name into its components.

        Args:
            toolgroup_name: The toolgroup name to parse (e.g. "builtin::rag/knowledge_search")

        Returns:
            A tuple of (tool_type, tool_group, tool_name)
        """
        split_names = toolgroup_name_with_maybe_tool_name.split("/")
        if len(split_names) == 2:
            # e.g. "builtin::rag"
            tool_group, tool_name = split_names
        else:
            tool_group, tool_name = split_names[0], None
        return tool_group, tool_name

    async def handle_documents(
        self,
        session_id: str,
        documents: List[Document],
        input_messages: List[Message],
        tool_defs: Dict[str, ToolDefinition],
    ) -> None:
        memory_tool = any(tool_def.tool_name == MEMORY_QUERY_TOOL for tool_def in tool_defs)
        code_interpreter_tool = any(tool_def.tool_name == BuiltinTool.code_interpreter for tool_def in tool_defs)
        content_items = []
        url_items = []
        pattern = re.compile("^(https?://|file://|data:)")
        for d in documents:
            if isinstance(d.content, URL):
                url_items.append(d.content)
            elif pattern.match(d.content):
                url_items.append(URL(uri=d.content))
            else:
                content_items.append(d)

        # Save the contents to a tempdir and use its path as a URL if code interpreter is present
        if code_interpreter_tool:
            for c in content_items:
                temp_file_path = os.path.join(self.tempdir, f"{make_random_string()}.txt")
                with open(temp_file_path, "w") as temp_file:
                    temp_file.write(c.content)
                url_items.append(URL(uri=f"file://{temp_file_path}"))

        if memory_tool and code_interpreter_tool:
            # if both memory and code_interpreter are available, we download the URLs
            # and attach the data to the last message.
            msg = await attachment_message(self.tempdir, url_items)
            input_messages.append(msg)
            # Since memory is present, add all the data to the memory bank
            await self.add_to_session_vector_db(session_id, documents)
        elif code_interpreter_tool:
            # if only code_interpreter is available, we download the URLs to a tempdir
            # and attach the path to them as a message to inference with the
            # assumption that the model invokes the code_interpreter tool with the path
            msg = await attachment_message(self.tempdir, url_items)
            input_messages.append(msg)
        elif memory_tool:
            # if only memory is available, we load the data from the URLs and content items to the memory bank
            await self.add_to_session_vector_db(session_id, documents)
        else:
            # if no memory or code_interpreter tool is available,
            # we try to load the data from the URLs and content items as a message to inference
            # and add it to the last message's context
            input_messages[-1].context = "\n".join(
                [doc.content for doc in content_items] + await load_data_from_urls(url_items)
            )

    async def _ensure_vector_db(self, session_id: str) -> str:
        session_info = await self.storage.get_session_info(session_id)
        if session_info is None:
            raise ValueError(f"Session {session_id} not found")

        if session_info.vector_db_id is None:
            vector_db_id = f"vector_db_{session_id}"

            # TODO: the semantic for registration is definitely not "creation"
            # so we need to fix it if we expect the agent to create a new vector db
            # for each session
            await self.vector_io_api.register_vector_db(
                vector_db_id=vector_db_id,
                embedding_model="all-MiniLM-L6-v2",
            )
            await self.storage.add_vector_db_to_session(session_id, vector_db_id)
        else:
            vector_db_id = session_info.vector_db_id

        return vector_db_id

    async def add_to_session_vector_db(self, session_id: str, data: List[Document]) -> None:
        vector_db_id = await self._ensure_vector_db(session_id)
        documents = [
            RAGDocument(
                document_id=str(uuid.uuid4()),
                content=a.content,
                mime_type=a.mime_type,
                metadata={},
            )
            for a in data
        ]
        await self.tool_runtime_api.rag_tool.insert(
            documents=documents,
            vector_db_id=vector_db_id,
            chunk_size_in_tokens=512,
        )


async def load_data_from_urls(urls: List[URL]) -> List[str]:
    data = []
    for url in urls:
        uri = url.uri
        if uri.startswith("file://"):
            filepath = uri[len("file://") :]
            with open(filepath, "r") as f:
                data.append(f.read())
        elif uri.startswith("http"):
            async with httpx.AsyncClient() as client:
                r = await client.get(uri)
                resp = r.text
                data.append(resp)
    return data


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
                text=f'# User provided a file accessible to you at "{filepath}"\nYou can use code_interpreter to load and inspect it.'
            )
        )

    return ToolResponseMessage(
        call_id="",
        tool_name=BuiltinTool.code_interpreter,
        content=content,
    )


async def execute_tool_call_maybe(
    tool_runtime_api: ToolRuntime,
    session_id: str,
    tool_call: ToolCall,
    toolgroup_args: Dict[str, Dict[str, Any]],
    tool_to_group: Dict[str, str],
) -> ToolInvocationResult:
    name = tool_call.tool_name
    group_name = tool_to_group.get(name, None)
    if group_name is None:
        raise ValueError(f"Tool {name} not found in any tool group")
    # get the arguments generated by the model and augment with toolgroup arg overrides for the agent
    tool_call_args = tool_call.arguments
    tool_call_args.update(toolgroup_args.get(group_name, {}))
    if isinstance(name, BuiltinTool):
        if name == BuiltinTool.brave_search:
            name = WEB_SEARCH_TOOL
        else:
            name = name.value

    result = await tool_runtime_api.invoke_tool(
        tool_name=name,
        kwargs=dict(
            session_id=session_id,
            **tool_call_args,
        ),
    )
    return result


def _interpret_content_as_attachment(
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
