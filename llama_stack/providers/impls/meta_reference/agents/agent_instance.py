# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import copy
import os
import re
import secrets
import shutil
import string
import tempfile
import uuid
from datetime import datetime
from typing import AsyncGenerator, List, Tuple
from urllib.parse import urlparse

import httpx

from termcolor import cprint

from llama_stack.apis.agents import *  # noqa: F403
from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.apis.safety import *  # noqa: F403

from llama_stack.providers.utils.kvstore import KVStore
from llama_stack.providers.utils.telemetry import tracing

from .persistence import AgentPersistence
from .rag.context_retriever import generate_rag_query
from .safety import SafetyException, ShieldRunnerMixin
from .tools.base import BaseTool
from .tools.builtin import (
    CodeInterpreterTool,
    interpret_content_as_attachment,
    PhotogenTool,
    SearchTool,
    WolframAlphaTool,
)
from .tools.safety import SafeTool


def make_random_string(length: int = 8):
    return "".join(
        secrets.choice(string.ascii_letters + string.digits) for _ in range(length)
    )


class ChatAgent(ShieldRunnerMixin):
    def __init__(
        self,
        agent_id: str,
        agent_config: AgentConfig,
        inference_api: Inference,
        memory_api: Memory,
        safety_api: Safety,
        persistence_store: KVStore,
    ):
        self.agent_id = agent_id
        self.agent_config = agent_config
        self.inference_api = inference_api
        self.memory_api = memory_api
        self.safety_api = safety_api
        self.storage = AgentPersistence(agent_id, persistence_store)

        self.tempdir = tempfile.mkdtemp()

        builtin_tools = []
        for tool_defn in agent_config.tools:
            if isinstance(tool_defn, WolframAlphaToolDefinition):
                tool = WolframAlphaTool(tool_defn.api_key)
            elif isinstance(tool_defn, SearchToolDefinition):
                tool = SearchTool(tool_defn.engine, tool_defn.api_key)
            elif isinstance(tool_defn, CodeInterpreterToolDefinition):
                tool = CodeInterpreterTool()
            elif isinstance(tool_defn, PhotogenToolDefinition):
                tool = PhotogenTool(dump_dir=self.tempdir)
            else:
                continue

            builtin_tools.append(
                SafeTool(
                    tool,
                    safety_api,
                    tool_defn.input_shields,
                    tool_defn.output_shields,
                )
            )
        self.tools_dict = {t.get_name(): t for t in builtin_tools}

        ShieldRunnerMixin.__init__(
            self,
            safety_api,
            input_shields=agent_config.input_shields,
            output_shields=agent_config.output_shields,
        )

    def __del__(self):
        shutil.rmtree(self.tempdir)

    def turn_to_messages(self, turn: Turn) -> List[Message]:
        messages = []

        # We do not want to keep adding RAG context to the input messages
        # May be this should be a parameter of the agentic instance
        # that can define its behavior in a custom way
        for m in turn.input_messages:
            msg = m.copy()
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
        # print_dialog(messages)
        return messages

    async def create_session(self, name: str) -> str:
        return await self.storage.create_session(name)

    @tracing.span("create_and_execute_turn")
    async def create_and_execute_turn(
        self, request: AgentTurnCreateRequest
    ) -> AsyncGenerator:
        session_info = await self.storage.get_session_info(request.session_id)
        if session_info is None:
            raise ValueError(f"Session {request.session_id} not found")

        turns = await self.storage.get_session_turns(request.session_id)

        messages = []
        if len(turns) == 0 and self.agent_config.instructions != "":
            messages.append(SystemMessage(content=self.agent_config.instructions))

        for i, turn in enumerate(turns):
            messages.extend(self.turn_to_messages(turn))

        messages.extend(request.messages)

        turn_id = str(uuid.uuid4())
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
                cprint(
                    f"{chunk.role.capitalize()}: {chunk.content}",
                    "white",
                    attrs=["bold"],
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

        async for res in self.run_multiple_shields_wrapper(
            turn_id, messages, self.output_shields, "assistant-output"
        ):
            if isinstance(res, bool):
                return
            else:
                yield res

        yield final_response

    @tracing.span("run_shields")
    async def run_multiple_shields_wrapper(
        self,
        turn_id: str,
        messages: List[Message],
        shields: List[str],
        touchpoint: str,
    ) -> AsyncGenerator:
        if len(shields) == 0:
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

    async def _run(
        self,
        session_id: str,
        turn_id: str,
        input_messages: List[Message],
        attachments: List[Attachment],
        sampling_params: SamplingParams,
        stream: bool = False,
    ) -> AsyncGenerator:
        enabled_tools = set(t.type for t in self.agent_config.tools)
        need_rag_context = await self._should_retrieve_context(
            input_messages, attachments
        )
        if need_rag_context:
            step_id = str(uuid.uuid4())
            yield AgentTurnResponseStreamChunk(
                event=AgentTurnResponseEvent(
                    payload=AgentTurnResponseStepStartPayload(
                        step_type=StepType.memory_retrieval.value,
                        step_id=step_id,
                    )
                )
            )

            # TODO: find older context from the session and either replace it
            # or append with a sliding window. this is really a very simplistic implementation
            with tracing.span("retrieve_rag_context"):
                rag_context, bank_ids = await self._retrieve_context(
                    session_id, input_messages, attachments
                )

            step_id = str(uuid.uuid4())
            yield AgentTurnResponseStreamChunk(
                event=AgentTurnResponseEvent(
                    payload=AgentTurnResponseStepCompletePayload(
                        step_type=StepType.memory_retrieval.value,
                        step_id=step_id,
                        step_details=MemoryRetrievalStep(
                            turn_id=turn_id,
                            step_id=step_id,
                            memory_bank_ids=bank_ids,
                            inserted_context=rag_context or "",
                        ),
                    )
                )
            )

            if rag_context:
                last_message = input_messages[-1]
                last_message.context = "\n".join(rag_context)

        elif attachments and AgentTool.code_interpreter.value in enabled_tools:
            urls = [a.content for a in attachments if isinstance(a.content, URL)]
            # TODO: we need to migrate URL away from str type
            pattern = re.compile("^(https?://|file://|data:)")
            urls += [
                URL(uri=a.content) for a in attachments if pattern.match(a.content)
            ]
            msg = await attachment_message(self.tempdir, urls)
            input_messages.append(msg)

        output_attachments = []

        n_iter = 0
        while True:
            msg = input_messages[-1]
            if msg.role == Role.user.value:
                color = "blue"
            elif msg.role == Role.ipython.value:
                color = "yellow"
            else:
                color = None
            if len(str(msg)) > 1000:
                msg_str = f"{str(msg)[:500]}...<more>...{str(msg)[-500:]}"
            else:
                msg_str = str(msg)
            cprint(f"{msg_str}", color=color)

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

            with tracing.span("inference"):
                async for chunk in self.inference_api.chat_completion(
                    self.agent_config.model,
                    input_messages,
                    tools=self._get_tools(),
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
                                        model_response_text_delta="",
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
                                        model_response_text_delta=event.delta,
                                    )
                                )
                            )
                    else:
                        raise ValueError(f"Unexpected delta type {type(delta)}")

                    if event.stop_reason is not None:
                        stop_reason = event.stop_reason

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
                cprint("Done with MAX iterations, exiting.")
                yield message
                break

            if stop_reason == StopReason.out_of_tokens:
                cprint("Out of token budget, exiting.")
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

                    with tracing.span("tool_execution"):
                        result_messages = await execute_tool_call_maybe(
                            self.tools_dict,
                            [message],
                        )
                        assert (
                            len(result_messages) == 1
                        ), "Currently not supporting multiple messages"
                        result_message = result_messages[0]

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
                    yield AgentTurnResponseStreamChunk(
                        event=AgentTurnResponseEvent(
                            payload=AgentTurnResponseStepCompletePayload(
                                step_type=StepType.shield_call.value,
                                step_details=ShieldCallStep(
                                    step_id=str(uuid.uuid4()),
                                    turn_id=turn_id,
                                    violation=None,
                                ),
                            )
                        )
                    )

                except SafetyException as e:
                    yield AgentTurnResponseStreamChunk(
                        event=AgentTurnResponseEvent(
                            payload=AgentTurnResponseStepCompletePayload(
                                step_type=StepType.shield_call.value,
                                step_details=ShieldCallStep(
                                    step_id=str(uuid.uuid4()),
                                    turn_id=turn_id,
                                    violation=e.violation,
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

                if out_attachment := interpret_content_as_attachment(
                    result_message.content
                ):
                    # NOTE: when we push this message back to the model, the model may ignore the
                    # attached file path etc. since the model is trained to only provide a user message
                    # with the summary. We keep all generated attachments and then attach them to final message
                    output_attachments.append(out_attachment)

                input_messages = input_messages + [message, result_message]

            n_iter += 1

    async def _ensure_memory_bank(self, session_id: str) -> str:
        session_info = await self.storage.get_session_info(session_id)
        if session_info is None:
            raise ValueError(f"Session {session_id} not found")

        if session_info.memory_bank_id is None:
            memory_bank = await self.memory_api.create_memory_bank(
                name=f"memory_bank_{session_id}",
                config=VectorMemoryBankConfig(
                    embedding_model="all-MiniLM-L6-v2",
                    chunk_size_in_tokens=512,
                ),
            )
            bank_id = memory_bank.bank_id
            await self.storage.add_memory_bank_to_session(session_id, bank_id)
        else:
            bank_id = session_info.memory_bank_id

        return bank_id

    async def _should_retrieve_context(
        self, messages: List[Message], attachments: List[Attachment]
    ) -> bool:
        enabled_tools = set(t.type for t in self.agent_config.tools)
        if attachments:
            if (
                AgentTool.code_interpreter.value in enabled_tools
                and self.agent_config.tool_choice == ToolChoice.required
            ):
                return False
            else:
                return True

        return AgentTool.memory.value in enabled_tools

    def _memory_tool_definition(self) -> Optional[MemoryToolDefinition]:
        for t in self.agent_config.tools:
            if t.type == AgentTool.memory.value:
                return t

        return None

    async def _retrieve_context(
        self, session_id: str, messages: List[Message], attachments: List[Attachment]
    ) -> Tuple[List[str], List[int]]:  # (rag_context, bank_ids)
        bank_ids = []

        memory = self._memory_tool_definition()
        assert memory is not None, "Memory tool not configured"
        bank_ids.extend(c.bank_id for c in memory.memory_bank_configs)

        if attachments:
            bank_id = await self._ensure_memory_bank(session_id)
            bank_ids.append(bank_id)

            documents = [
                MemoryBankDocument(
                    document_id=str(uuid.uuid4()),
                    content=a.content,
                    mime_type=a.mime_type,
                    metadata={},
                )
                for a in attachments
            ]
            with tracing.span("insert_documents"):
                await self.memory_api.insert_documents(bank_id, documents)
        else:
            session_info = await self.storage.get_session_info(session_id)
            if session_info.memory_bank_id:
                bank_ids.append(session_info.memory_bank_id)

        if not bank_ids:
            # this can happen if the per-session memory bank is not yet populated
            # (i.e., no prior turns uploaded an Attachment)
            return None, []

        query = await generate_rag_query(
            memory.query_generator_config, messages, inference_api=self.inference_api
        )
        tasks = [
            self.memory_api.query_documents(
                bank_id=bank_id,
                query=query,
                params={
                    "max_chunks": 5,
                },
            )
            for bank_id in bank_ids
        ]
        results: List[QueryDocumentsResponse] = await asyncio.gather(*tasks)
        chunks = [c for r in results for c in r.chunks]
        scores = [s for r in results for s in r.scores]

        # sort by score
        chunks, scores = zip(
            *sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        )
        if not chunks:
            return None, bank_ids

        tokens = 0
        picked = []
        for c in chunks[: memory.max_chunks]:
            tokens += c.token_count
            if tokens > memory.max_tokens_in_context:
                cprint(
                    f"Using {len(picked)} chunks; reached max tokens in context: {tokens}",
                    "red",
                )
                break
            picked.append(f"id:{c.document_id}; content:{c.content}")

        return [
            "Here are the retrieved documents for relevant context:\n=== START-RETRIEVED-CONTEXT ===\n",
            *picked,
            "\n=== END-RETRIEVED-CONTEXT ===\n",
        ], bank_ids

    def _get_tools(self) -> List[ToolDefinition]:
        ret = []
        for t in self.agent_config.tools:
            if isinstance(t, SearchToolDefinition):
                ret.append(ToolDefinition(tool_name=BuiltinTool.brave_search))
            elif isinstance(t, WolframAlphaToolDefinition):
                ret.append(ToolDefinition(tool_name=BuiltinTool.wolfram_alpha))
            elif isinstance(t, PhotogenToolDefinition):
                ret.append(ToolDefinition(tool_name=BuiltinTool.photogen))
            elif isinstance(t, CodeInterpreterToolDefinition):
                ret.append(ToolDefinition(tool_name=BuiltinTool.code_interpreter))
            elif isinstance(t, FunctionCallToolDefinition):
                ret.append(
                    ToolDefinition(
                        tool_name=t.function_name,
                        description=t.description,
                        parameters=t.parameters,
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
            print(f"Downloading {url} -> {filepath}")

            async with httpx.AsyncClient() as client:
                r = await client.get(uri)
                resp = r.text
                with open(filepath, "w") as fp:
                    fp.write(resp)
        else:
            raise ValueError(f"Unsupported URL {url}")

        content.append(f'# There is a file accessible to you at "{filepath}"\n')

    return ToolResponseMessage(
        call_id="",
        tool_name=BuiltinTool.code_interpreter,
        content=content,
    )


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
