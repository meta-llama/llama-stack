# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import copy
import uuid
from datetime import datetime
from typing import AsyncGenerator, List, Optional

from llama_toolchain.agentic_system.api.datatypes import (
    AgenticSystemInstanceConfig,
    AgenticSystemTurnResponseEvent,
    AgenticSystemTurnResponseEventType,
    AgenticSystemTurnResponseStepCompletePayload,
    AgenticSystemTurnResponseStepProgressPayload,
    AgenticSystemTurnResponseStepStartPayload,
    AgenticSystemTurnResponseTurnCompletePayload,
    AgenticSystemTurnResponseTurnStartPayload,
    InferenceStep,
    Session,
    ShieldCallStep,
    StepType,
    ToolExecutionStep,
    Turn,
)

from llama_toolchain.inference.api import ChatCompletionRequest, Inference

from llama_toolchain.inference.api.datatypes import (
    Attachment,
    BuiltinTool,
    ChatCompletionResponseEventType,
    CompletionMessage,
    Message,
    Role,
    SamplingParams,
    StopReason,
    ToolCallDelta,
    ToolCallParseStatus,
    ToolDefinition,
    ToolResponse,
    ToolResponseMessage,
    URL,
)
from llama_toolchain.safety.api import Safety
from llama_toolchain.safety.api.datatypes import (
    BuiltinShield,
    ShieldDefinition,
    ShieldResponse,
)
from termcolor import cprint
from llama_toolchain.agentic_system.api.endpoints import *  # noqa

from .safety import SafetyException, ShieldRunnerMixin
from .system_prompt import get_agentic_prefix_messages
from .tools.base import BaseTool
from .tools.builtin import SingleMessageBuiltinTool


class AgentInstance(ShieldRunnerMixin):
    def __init__(
        self,
        system_id: int,
        instance_config: AgenticSystemInstanceConfig,
        model: str,
        inference_api: Inference,
        safety_api: Safety,
        builtin_tools: List[SingleMessageBuiltinTool],
        custom_tool_definitions: List[ToolDefinition],
        input_shields: List[ShieldDefinition],
        output_shields: List[ShieldDefinition],
        max_infer_iters: int = 10,
        prefix_messages: Optional[List[Message]] = None,
    ):
        self.system_id = system_id
        self.instance_config = instance_config

        self.model = model
        self.inference_api = inference_api
        self.safety_api = safety_api

        if prefix_messages is not None and len(prefix_messages) > 0:
            self.prefix_messages = prefix_messages
        else:
            self.prefix_messages = get_agentic_prefix_messages(
                builtin_tools, custom_tool_definitions
            )

        for m in self.prefix_messages:
            print(m.content)

        self.max_infer_iters = max_infer_iters
        self.tools_dict = {t.get_name(): t for t in builtin_tools}

        self.sessions = {}

        ShieldRunnerMixin.__init__(
            self,
            safety_api,
            input_shields=input_shields,
            output_shields=output_shields,
        )

    def create_session(self, name: str) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            session_name=name,
            turns=[],
            started_at=datetime.now(),
        )
        self.sessions[session_id] = session
        return session

    async def create_and_execute_turn(
        self, request: AgenticSystemTurnCreateRequest
    ) -> AsyncGenerator:
        assert (
            request.session_id in self.sessions
        ), f"Session {request.session_id} not found"

        session = self.sessions[request.session_id]

        messages = []
        for i, turn in enumerate(session.turns):
            # print(f"turn {i}")
            # print_dialog(turn.input_messages)
            messages.extend(turn.input_messages)
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
                    response = step.response
                    if response.is_violation:
                        # TODO: Properly persist the
                        # CompletionMessage itself in the ShieldResponse
                        messages.append(
                            CompletionMessage(
                                content=response.violation_return_message,
                                stop_reason=StopReason.end_of_turn,
                            )
                        )

        messages.extend(request.messages)

        # print("processed dialog ======== ")
        # print_dialog(messages)

        turn_id = str(uuid.uuid4())
        params = self.instance_config.sampling_params
        start_time = datetime.now()
        yield AgenticSystemTurnResponseStreamChunk(
            event=AgenticSystemTurnResponseEvent(
                payload=AgenticSystemTurnResponseTurnStartPayload(
                    turn_id=turn_id,
                )
            )
        )

        steps = []
        output_message = None
        async for chunk in self.run(
            turn_id=turn_id,
            input_messages=messages,
            temperature=params.temperature,
            top_p=params.top_p,
            stream=request.stream,
            max_gen_len=params.max_tokens,
        ):
            if isinstance(chunk, CompletionMessage):
                cprint(
                    f"{chunk.role.capitalize()}: {chunk.content}",
                    "white",
                    attrs=["bold"],
                )
                output_message = chunk
                continue

            assert isinstance(
                chunk, AgenticSystemTurnResponseStreamChunk
            ), f"Unexpected type {type(chunk)}"
            event = chunk.event
            if (
                event.payload.event_type
                == AgenticSystemTurnResponseEventType.step_complete.value
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
        session.turns.append(turn)

        chunk = AgenticSystemTurnResponseStreamChunk(
            event=AgenticSystemTurnResponseEvent(
                payload=AgenticSystemTurnResponseTurnCompletePayload(
                    turn=turn,
                )
            )
        )
        yield chunk

    async def run_shields_wrapper(
        self,
        turn_id: str,
        messages: List[Message],
        shields: List[ShieldDefinition],
        touchpoint: str,
    ) -> AsyncGenerator:
        if len(shields) == 0:
            return

        step_id = str(uuid.uuid4())
        try:
            yield AgenticSystemTurnResponseStreamChunk(
                event=AgenticSystemTurnResponseEvent(
                    payload=AgenticSystemTurnResponseStepStartPayload(
                        step_type=StepType.shield_call.value,
                        step_id=step_id,
                        metadata=dict(touchpoint=touchpoint),
                    )
                )
            )
            await self.run_shields(messages, shields)

        except SafetyException as e:

            yield AgenticSystemTurnResponseStreamChunk(
                event=AgenticSystemTurnResponseEvent(
                    payload=AgenticSystemTurnResponseStepCompletePayload(
                        step_type=StepType.shield_call.value,
                        step_details=ShieldCallStep(
                            step_id=step_id,
                            turn_id=turn_id,
                            response=e.response,
                        ),
                    )
                )
            )

            yield CompletionMessage(
                content=str(e),
                stop_reason=StopReason.end_of_turn,
            )
            yield False

        yield AgenticSystemTurnResponseStreamChunk(
            event=AgenticSystemTurnResponseEvent(
                payload=AgenticSystemTurnResponseStepCompletePayload(
                    step_type=StepType.shield_call.value,
                    step_details=ShieldCallStep(
                        step_id=step_id,
                        turn_id=turn_id,
                        response=ShieldResponse(
                            # TODO: fix this, give each shield a shield type method and
                            # fire one event for each shield run
                            shield_type=BuiltinShield.llama_guard,
                            is_violation=False,
                        ),
                    ),
                )
            )
        )

    async def run(
        self,
        turn_id: str,
        input_messages: List[Message],
        temperature: float,
        top_p: float,
        stream: bool = False,
        max_gen_len: Optional[int] = None,
    ) -> AsyncGenerator:
        # Doing async generators makes downstream code much simpler and everything amenable to
        # stremaing. However, it also makes things complicated here because AsyncGenerators cannot
        # return a "final value" for the `yield from` statement. we simulate that by yielding a
        # final boolean (to see whether an exception happened) and then explicitly testing for it.

        async for res in self.run_shields_wrapper(
            turn_id, input_messages, self.input_shields, "user-input"
        ):
            if isinstance(res, bool):
                return
            else:
                yield res

        async for res in self._run(
            turn_id, input_messages, temperature, top_p, stream, max_gen_len
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

        async for res in self.run_shields_wrapper(
            turn_id, messages, self.output_shields, "assistant-output"
        ):
            if isinstance(res, bool):
                return
            else:
                yield res

        yield final_response

    async def _run(
        self,
        turn_id: str,
        input_messages: List[Message],
        temperature: float,
        top_p: float,
        stream: bool = False,
        max_gen_len: Optional[int] = None,
    ) -> AsyncGenerator:
        input_messages = preprocess_dialog(input_messages, self.prefix_messages)

        attachments = []

        n_iter = 0
        while True:
            msg = input_messages[-1]
            if msg.role == Role.user.value:
                color = "blue"
            elif msg.role == Role.ipython.value:
                color = "yellow"
            else:
                color = None
            cprint(f"{str(msg)}", color=color)

            step_id = str(uuid.uuid4())
            yield AgenticSystemTurnResponseStreamChunk(
                event=AgenticSystemTurnResponseEvent(
                    payload=AgenticSystemTurnResponseStepStartPayload(
                        step_type=StepType.inference.value,
                        step_id=step_id,
                    )
                )
            )

            # where are the available tools?
            req = ChatCompletionRequest(
                model=self.model,
                messages=input_messages,
                available_tools=self.instance_config.available_tools,
                stream=True,
                sampling_params=SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_gen_len,
                ),
            )

            tool_calls = []
            content = ""
            stop_reason = None
            async for chunk in self.inference_api.chat_completion(req):
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
                        yield AgenticSystemTurnResponseStreamChunk(
                            event=AgenticSystemTurnResponseEvent(
                                payload=AgenticSystemTurnResponseStepProgressPayload(
                                    step_type=StepType.inference.value,
                                    step_id=step_id,
                                    model_response_text_delta="",
                                    tool_call_delta=delta,
                                )
                            )
                        )

                elif isinstance(delta, str):
                    content += delta
                    if stream and event.stop_reason is None:
                        yield AgenticSystemTurnResponseStreamChunk(
                            event=AgenticSystemTurnResponseEvent(
                                payload=AgenticSystemTurnResponseStepProgressPayload(
                                    step_type=StepType.inference.value,
                                    step_id=step_id,
                                    model_response_text_delta=event.delta,
                                )
                            )
                        )
                else:
                    raise ValueError(f"Unexpected delta type {type(delta)}")

                if event.stop_reason is not None:
                    stop_reason = event.stop_reason

            stop_reason = stop_reason or StopReason.out_of_tokens
            message = CompletionMessage(
                content=content,
                stop_reason=stop_reason,
                tool_calls=tool_calls,
            )

            yield AgenticSystemTurnResponseStreamChunk(
                event=AgenticSystemTurnResponseEvent(
                    payload=AgenticSystemTurnResponseStepCompletePayload(
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

            if n_iter >= self.max_infer_iters:
                cprint("Done with MAX iterations, exiting.")
                yield message
                break

            if stop_reason == StopReason.out_of_tokens:
                cprint("Out of token budget, exiting.")
                yield message
                break

            if len(message.tool_calls) == 0:
                if stop_reason == StopReason.end_of_turn:
                    if len(attachments) > 0:
                        if isinstance(message.content, list):
                            message.content += attachments
                        else:
                            message.content = [message.content] + attachments
                    yield message
                else:
                    cprint(f"Partial message: {str(message)}", color="green")
                    input_messages = input_messages + [message]
            else:
                cprint(f"{str(message)}", color="green")
                try:
                    tool_call = message.tool_calls[0]

                    name = tool_call.tool_name
                    if not isinstance(name, BuiltinTool):
                        yield message
                        return

                    step_id = str(uuid.uuid4())
                    yield AgenticSystemTurnResponseStreamChunk(
                        event=AgenticSystemTurnResponseEvent(
                            payload=AgenticSystemTurnResponseStepStartPayload(
                                step_type=StepType.tool_execution.value,
                                step_id=step_id,
                            )
                        )
                    )
                    yield AgenticSystemTurnResponseStreamChunk(
                        event=AgenticSystemTurnResponseEvent(
                            payload=AgenticSystemTurnResponseStepProgressPayload(
                                step_type=StepType.tool_execution.value,
                                step_id=step_id,
                                tool_call=tool_call,
                            )
                        )
                    )

                    result_messages = await execute_tool_call_maybe(
                        self.tools_dict,
                        [message],
                    )
                    assert (
                        len(result_messages) == 1
                    ), "Currently not supporting multiple messages"
                    result_message = result_messages[0]

                    yield AgenticSystemTurnResponseStreamChunk(
                        event=AgenticSystemTurnResponseEvent(
                            payload=AgenticSystemTurnResponseStepCompletePayload(
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
                    yield AgenticSystemTurnResponseStreamChunk(
                        event=AgenticSystemTurnResponseEvent(
                            payload=AgenticSystemTurnResponseStepCompletePayload(
                                step_type=StepType.shield_call.value,
                                step_details=ShieldCallStep(
                                    step_id=str(uuid.uuid4()),
                                    turn_id=turn_id,
                                    response=ShieldResponse(
                                        # TODO: fix this, give each shield a shield type method and
                                        # fire one event for each shield run
                                        shield_type=BuiltinShield.llama_guard,
                                        is_violation=False,
                                    ),
                                ),
                            )
                        )
                    )

                except SafetyException as e:
                    yield AgenticSystemTurnResponseStreamChunk(
                        event=AgenticSystemTurnResponseEvent(
                            payload=AgenticSystemTurnResponseStepCompletePayload(
                                step_type=StepType.shield_call.value,
                                step_details=ShieldCallStep(
                                    step_id=str(uuid.uuid4()),
                                    turn_id=turn_id,
                                    response=e.response,
                                ),
                            )
                        )
                    )

                    yield CompletionMessage(
                        content=str(e),
                        stop_reason=StopReason.end_of_turn,
                    )
                    yield False
                    return

                if isinstance(result_message.content, Attachment):
                    # NOTE: when we push this message back to the model, the model may ignore the
                    # attached file path etc. since the model is trained to only provide a user message
                    # with the summary. We keep all generated attachments and then attach them to final message
                    attachments.append(result_message.content)
                elif isinstance(result_message.content, list) or isinstance(
                    result_message.content, tuple
                ):
                    for c in result_message.content:
                        if isinstance(c, Attachment):
                            attachments.append(c)

                input_messages = input_messages + [message, result_message]

            n_iter += 1


def attachment_message(url: URL) -> ToolResponseMessage:
    uri = url.uri
    assert uri.startswith("file://")
    filepath = uri[len("file://") :]

    return ToolResponseMessage(
        call_id="",
        tool_name=BuiltinTool.code_interpreter,
        content=f'# There is a file accessible to you at "{filepath}"',
    )


def preprocess_dialog(
    messages: List[Message], prefix_messages: List[Message]
) -> List[Message]:
    """
    Preprocesses the dialog by removing the system message and
    adding the system message to the beginning of the dialog.
    """
    ret = prefix_messages.copy()

    for m in messages:
        if m.role == Role.system.value:
            continue

        # NOTE: the ideal behavior is to use `file_path = ...` but that
        # means we need to have stateful execution o    f code which we currently
        # do not have.
        if isinstance(m.content, Attachment):
            ret.append(attachment_message(m.content.url))
        elif isinstance(m.content, list):
            for c in m.content:
                if isinstance(c, Attachment):
                    ret.append(attachment_message(c.url))

        ret.append(m)

    return ret


async def execute_tool_call_maybe(
    tools_dict: Dict[str, BaseTool], messages: List[CompletionMessage]
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
    assert isinstance(name, BuiltinTool)

    name = name.value

    assert name in tools_dict, f"Tool {name} not found"
    tool = tools_dict[name]
    result_messages = await tool.run(messages)
    return result_messages


def print_dialog(messages: List[Message]):
    for i, m in enumerate(messages):
        if m.role == Role.user.value:
            color = "red"
        elif m.role == Role.assistant.value:
            color = "white"
        elif m.role == Role.ipython.value:
            color = "yellow"
        elif m.role == Role.system.value:
            color = "green"
        else:
            color = "white"

        s = str(m)
        cprint(f"{i} ::: {s[:100]}...", color=color)
