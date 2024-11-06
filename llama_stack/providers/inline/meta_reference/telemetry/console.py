# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from llama_stack.apis.telemetry import *  # noqa: F403
from .config import ConsoleConfig


class ConsoleTelemetryImpl(Telemetry):
    def __init__(self, config: ConsoleConfig) -> None:
        self.config = config
        self.spans = {}

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def log_event(self, event: Event):
        if (
            isinstance(event, StructuredLogEvent)
            and event.payload.type == StructuredLogType.SPAN_START.value
        ):
            self.spans[event.span_id] = event.payload

        names = []
        span_id = event.span_id
        while True:
            span_payload = self.spans.get(span_id)
            if not span_payload:
                break

            names = [span_payload.name] + names
            span_id = span_payload.parent_span_id

        span_name = ".".join(names) if names else None

        formatted = format_event(event, span_name)
        if formatted:
            print(formatted)

    async def get_trace(self, trace_id: str) -> Trace:
        raise NotImplementedError()


COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}

SEVERITY_COLORS = {
    LogSeverity.VERBOSE: COLORS["dim"] + COLORS["white"],
    LogSeverity.DEBUG: COLORS["cyan"],
    LogSeverity.INFO: COLORS["green"],
    LogSeverity.WARN: COLORS["yellow"],
    LogSeverity.ERROR: COLORS["red"],
    LogSeverity.CRITICAL: COLORS["bold"] + COLORS["red"],
}


def format_event(event: Event, span_name: str) -> Optional[str]:
    timestamp = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
    span = ""
    if span_name:
        span = f"{COLORS['magenta']}[{span_name}]{COLORS['reset']} "
    if isinstance(event, UnstructuredLogEvent):
        severity_color = SEVERITY_COLORS.get(event.severity, COLORS["reset"])
        return (
            f"{COLORS['dim']}{timestamp}{COLORS['reset']} "
            f"{severity_color}[{event.severity.name}]{COLORS['reset']} "
            f"{span}"
            f"{event.message}"
        )

    elif isinstance(event, StructuredLogEvent):
        return None

    return f"Unknown event type: {event}"
