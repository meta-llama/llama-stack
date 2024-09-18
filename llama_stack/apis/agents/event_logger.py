# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_models.llama3.api.tool_utils import ToolUtils

from llama_stack.apis.agents import AgentTurnResponseEventType, StepType

from termcolor import cprint


class LogEvent:
    def __init__(
        self,
        role: Optional[str] = None,
        content: str = "",
        end: str = "\n",
        color="white",
    ):
        self.role = role
        self.content = content
        self.color = color
        self.end = "\n" if end is None else end

    def __str__(self):
        if self.role is not None:
            return f"{self.role}> {self.content}"
        else:
            return f"{self.content}"

    def print(self, flush=True):
        cprint(f"{str(self)}", color=self.color, end=self.end, flush=flush)


EventType = AgentTurnResponseEventType


class EventLogger:
    async def log(
        self,
        event_generator,
        stream=True,
        tool_prompt_format: ToolPromptFormat = ToolPromptFormat.json,
    ):
        previous_event_type = None
        previous_step_type = None

        async for chunk in event_generator:
            if not hasattr(chunk, "event"):
                # Need to check for custom tool first
                # since it does not produce event but instead
                # a Message
                if isinstance(chunk, ToolResponseMessage):
                    yield chunk, LogEvent(
                        role="CustomTool", content=chunk.content, color="grey"
                    )
                continue

            event = chunk.event
            event_type = event.payload.event_type
            if event_type in {
                EventType.turn_start.value,
                EventType.turn_complete.value,
            }:
                # Currently not logging any turn realted info
                yield event, None
                continue

            step_type = event.payload.step_type
            # handle safety
            if (
                step_type == StepType.shield_call
                and event_type == EventType.step_complete.value
            ):
                response = event.payload.step_details.response
                if not response.is_violation:
                    yield event, LogEvent(
                        role=step_type, content="No Violation", color="magenta"
                    )
                else:
                    yield event, LogEvent(
                        role=step_type,
                        content=f"{response.violation_type} {response.violation_return_message}",
                        color="red",
                    )

            # handle inference
            if step_type == StepType.inference:
                if stream:
                    if event_type == EventType.step_start.value:
                        # TODO: Currently this event is never received
                        yield event, LogEvent(
                            role=step_type, content="", end="", color="yellow"
                        )
                    elif event_type == EventType.step_progress.value:
                        # HACK: if previous was not step/event was not inference's step_progress
                        # this is the first time we are getting model inference response
                        # aka equivalent to step_start for inference. Hence,
                        # start with "Model>".
                        if (
                            previous_event_type != EventType.step_progress.value
                            and previous_step_type != StepType.inference
                        ):
                            yield event, LogEvent(
                                role=step_type, content="", end="", color="yellow"
                            )

                        if event.payload.tool_call_delta:
                            if isinstance(event.payload.tool_call_delta.content, str):
                                yield event, LogEvent(
                                    role=None,
                                    content=event.payload.tool_call_delta.content,
                                    end="",
                                    color="cyan",
                                )
                        else:
                            yield event, LogEvent(
                                role=None,
                                content=event.payload.model_response_text_delta,
                                end="",
                                color="yellow",
                            )
                    else:
                        # step_complete
                        yield event, LogEvent(role=None, content="")

                else:
                    # Not streaming
                    if event_type == EventType.step_complete.value:
                        response = event.payload.step_details.model_response
                        if response.tool_calls:
                            content = ToolUtils.encode_tool_call(
                                response.tool_calls[0], tool_prompt_format
                            )
                        else:
                            content = response.content
                        yield event, LogEvent(
                            role=step_type,
                            content=content,
                            color="yellow",
                        )

            # handle tool_execution
            if (
                step_type == StepType.tool_execution
                and
                # Only print tool calls and responses at the step_complete event
                event_type == EventType.step_complete.value
            ):
                details = event.payload.step_details
                for t in details.tool_calls:
                    yield event, LogEvent(
                        role=step_type,
                        content=f"Tool:{t.tool_name} Args:{t.arguments}",
                        color="green",
                    )
                for r in details.tool_responses:
                    yield event, LogEvent(
                        role=step_type,
                        content=f"Tool:{r.tool_name} Response:{r.content}",
                        color="green",
                    )

            if (
                step_type == StepType.memory_retrieval
                and event_type == EventType.step_complete.value
            ):
                details = event.payload.step_details
                content = interleaved_text_media_as_str(details.inserted_context)
                content = content[:200] + "..." if len(content) > 200 else content

                yield event, LogEvent(
                    role=step_type,
                    content=f"Retrieved context from banks: {details.memory_bank_ids}.\n====\n{content}\n>",
                    color="cyan",
                )

            preivous_event_type = event_type
            previous_step_type = step_type
