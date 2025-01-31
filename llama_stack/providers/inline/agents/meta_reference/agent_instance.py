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
from llama_models.llama3.api.datatypes import BuiltinTool, ToolCall, ToolParamDefinition
from pydantic import TypeAdapter

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
    AgentTurnResponseTurnCompletePayload,
    AgentTurnResponseTurnStartPayload,
    Attachment,
    Document,
    InferenceStep,
    ShieldCallStep,
    StepType,
    ToolExecutionStep,
    Turn,
)
from llama_stack.apis.common.content_types import (
    TextContentItem,
    ToolCallDelta,
    ToolCallParseStatus,
    URL,
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
from llama_stack.apis.tools import RAGDocument, RAGQueryConfig, ToolGroups, ToolRuntime
from llama_stack.apis.vector_io import VectorIO
from llama_stack.providers.utils.kvstore import KVStore
from llama_stack.providers.utils.memory.vector_store import concat_interleaved_content
from llama_stack.providers.utils.telemetry import tracing

from .persistence import AgentPersistence
from .safety import SafetyException, ShieldRunnerMixin

log = logging.getLogger(__name__)


def make_random_string(length: int = 8):
    return "".join(
        secrets.choice(string.ascii_letters + string.digits) for _ in range(length)
    )


TOOLS_ATTACHMENT_KEY_REGEX = re.compile(r"__tools_attachment__=(\{.*?\})")
MEMORY_QUERY_TOOL = "query_from_memory"
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
        for toolgroup in self.agent_config.toolgroups:
            if isinstance(toolgroup, AgentToolGroupWithArgs):
                toolgroups.add(toolgroup.name)
                toolgroup_args[toolgroup.name] = toolgroup.args
            else:
                toolgroups.add(toolgroup)
        if toolgroups_for_turn:
            for toolgroup in toolgroups_for_turn:
                if isinstance(toolgroup, AgentToolGroupWithArgs):
                    toolgroups.add(toolgroup.name)
                    toolgroup_args[toolgroup.name] = toolgroup.args
                else:
                    toolgroups.add(toolgroup)

        tool_defs, tool_to_group = await self._get_tool_defs(toolgroups_for_turn)
        if documents:
            await self.handle_documents(
                session_id, documents, input_messages, tool_defs
            )

        if RAG_TOOL_GROUP in toolgroups and len(input_messages) > 0:
            with tracing.span(MEMORY_QUERY_TOOL) as span:
                step_id = str(uuid.uuid4())
                yield AgentTurnResponseStreamChunk(
                    event=AgentTurnResponseEvent(
                        payload=AgentTurnResponseStepStartPayload(
                            step_type=StepType.tool_execution.value,
                            step_id=step_id,
                        )
                    )
                )

                args = toolgroup_args.get(RAG_TOOL_GROUP, {})
                vector_db_ids = args.get("vector_db_ids", [])
                query_config = args.get("query_config")
                if query_config:
                    query_config = TypeAdapter(RAGQueryConfig).validate_python(
                        query_config
                    )
                else:
                    # handle someone passing an empty dict
                    query_config = RAGQueryConfig()

                session_info = await self.storage.get_session_info(session_id)

                # if the session has a memory bank id, let the memory tool use it
                if session_info.vector_db_id:
                    vector_db_ids.append(session_info.vector_db_id)

                yield AgentTurnResponseStreamChunk(
                    event=AgentTurnResponseEvent(
                        payload=AgentTurnResponseStepProgressPayload(
                            step_type=StepType.tool_execution.value,
                            step_id=step_id,
                            delta=ToolCallDelta(
                                parse_status=ToolCallParseStatus.succeeded,
                                tool_call=ToolCall(
                                    call_id="",
                                    tool_name=MEMORY_QUERY_TOOL,
                                    arguments={},
                                ),
                            ),
                        )
                    )
                )
                result = await self.tool_runtime_api.rag_tool.query(
                    content=concat_interleaved_content(
                        [msg.content for msg in input_messages]
                    ),
                    vector_db_ids=vector_db_ids,
                    query_config=query_config,
                )
                retrieved_context = result.content

                yield AgentTurnResponseStreamChunk(
                    event=AgentTurnResponseEvent(
                        payload=AgentTurnResponseStepCompletePayload(
                            step_type=StepType.tool_execution.value,
                            step_id=step_id,
                            step_details=ToolExecutionStep(
                                step_id=step_id,
                                turn_id=turn_id,
                                tool_calls=[
                                    ToolCall(
                                        call_id="",
                                        tool_name=MEMORY_QUERY_TOOL,
                                        arguments={},
                                    )
                                ],
                                tool_responses=[
                                    ToolResponse(
                                        call_id="",
                                        tool_name=MEMORY_QUERY_TOOL,
                                        content=retrieved_context or [],
                                    )
                                ],
                            ),
                        )
                    )
                )
                span.set_attribute(
                    "input", [m.model_dump_json() for m in input_messages]
                )
                span.set_attribute("output", retrieved_context)
                span.set_attribute("tool_name", MEMORY_QUERY_TOOL)

                # append retrieved_context to the last user message
                for message in input_messages[::-1]:
                    if isinstance(message, UserMessage):
                        message.context = retrieved_context
                        break

        output_attachments = []

        n_iter = 0
        # Build a map of custom tools to their definitions for faster lookup
        client_tools = {}
        for tool in self.agent_config.client_tools:
            client_tools[tool.name] = tool
        while True:
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
                    tools=[
                        tool
                        for tool in tool_defs.values()
                        if tool_to_group.get(tool.tool_name, None) != RAG_TOOL_GROUP
                    ],
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
                            message.content += output_attachments
                        else:
                            message.content = [message.content] + output_attachments
                    yield message
                else:
                    log.info(f"Partial message: {str(message)}")
                    input_messages = input_messages + [message]
            else:
                log.info(f"{str(message)}")
                tool_call = message.tool_calls[0]
                if tool_call.tool_name in client_tools:
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
                            delta=ToolCallDelta(
                                parse_status=ToolCallParseStatus.in_progress,
                                tool_call=tool_call,
                            ),
                        )
                    )
                )

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
                    result_messages = await execute_tool_call_maybe(
                        self.tool_runtime_api,
                        session_id,
                        [message],
                        toolgroup_args,
                        tool_to_group,
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
                                    )
                                ],
                            ),
                        )
                    )
                )

                # TODO: add tool-input touchpoint and a "start" event for this step also
                # but that needs a lot more refactoring of Tool code potentially

                if out_attachment := _interpret_content_as_attachment(
                    result_message.content
                ):
                    # NOTE: when we push this message back to the model, the model may ignore the
                    # attached file path etc. since the model is trained to only provide a user message
                    # with the summary. We keep all generated attachments and then attach them to final message
                    output_attachments.append(out_attachment)

                input_messages = input_messages + [message, result_message]

            n_iter += 1

    async def _get_tool_defs(
        self, toolgroups_for_turn: Optional[List[AgentToolGroup]] = None
    ) -> Tuple[Dict[str, ToolDefinition], Dict[str, str]]:
        # Determine which tools to include
        agent_config_toolgroups = set(
            (
                toolgroup.name
                if isinstance(toolgroup, AgentToolGroupWithArgs)
                else toolgroup
            )
            for toolgroup in self.agent_config.toolgroups
        )
        toolgroups_for_turn_set = (
            agent_config_toolgroups
            if toolgroups_for_turn is None
            else {
                (
                    toolgroup.name
                    if isinstance(toolgroup, AgentToolGroupWithArgs)
                    else toolgroup
                )
                for toolgroup in toolgroups_for_turn
            }
        )

        tool_def_map = {}
        tool_to_group = {}

        for tool_def in self.agent_config.client_tools:
            if tool_def_map.get(tool_def.name, None):
                raise ValueError(f"Tool {tool_def.name} already exists")
            tool_def_map[tool_def.name] = ToolDefinition(
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
        for toolgroup_name in agent_config_toolgroups:
            if toolgroup_name not in toolgroups_for_turn_set:
                continue
            tools = await self.tool_groups_api.list_tools(toolgroup_id=toolgroup_name)
            for tool_def in tools.data:
                if (
                    toolgroup_name.startswith("builtin")
                    and toolgroup_name != RAG_TOOL_GROUP
                ):
                    tool_name = tool_def.identifier
                    built_in_type = BuiltinTool.brave_search
                    if tool_name == "web_search":
                        built_in_type = BuiltinTool.brave_search
                    else:
                        built_in_type = BuiltinTool(tool_name)

                    if tool_def_map.get(built_in_type, None):
                        raise ValueError(f"Tool {built_in_type} already exists")

                    tool_def_map[built_in_type] = ToolDefinition(
                        tool_name=built_in_type
                    )
                    tool_to_group[built_in_type] = tool_def.toolgroup_id
                    continue

                if tool_def_map.get(tool_def.identifier, None):
                    raise ValueError(f"Tool {tool_def.identifier} already exists")
                tool_def_map[tool_def.identifier] = ToolDefinition(
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

        return tool_def_map, tool_to_group

    async def handle_documents(
        self,
        session_id: str,
        documents: List[Document],
        input_messages: List[Message],
        tool_defs: Dict[str, ToolDefinition],
    ) -> None:
        memory_tool = tool_defs.get(MEMORY_QUERY_TOOL, None)
        code_interpreter_tool = tool_defs.get(BuiltinTool.code_interpreter, None)
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
                temp_file_path = os.path.join(
                    self.tempdir, f"{make_random_string()}.txt"
                )
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
                [doc.content for doc in content_items]
                + await load_data_from_urls(url_items)
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

    async def add_to_session_vector_db(
        self, session_id: str, data: List[Document]
    ) -> None:
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
                text=f'# There is a file accessible to you at "{filepath}"\n'
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
    messages: List[CompletionMessage],
    toolgroup_args: Dict[str, Dict[str, Any]],
    tool_to_group: Dict[str, str],
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

    return [
        ToolResponseMessage(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            content=result.content,
        )
    ]


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
