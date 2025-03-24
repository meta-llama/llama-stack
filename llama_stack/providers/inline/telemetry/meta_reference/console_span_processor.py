# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from datetime import datetime, timezone

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanProcessor
from opentelemetry.trace.status import StatusCode

# Colors for console output
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


class ConsoleSpanProcessor(SpanProcessor):
    def __init__(self, print_attributes: bool = False):
        self.print_attributes = print_attributes

    def on_start(self, span: ReadableSpan, parent_context=None) -> None:
        if span.attributes and span.attributes.get("__autotraced__"):
            return

        timestamp = datetime.fromtimestamp(span.start_time / 1e9, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]

        print(
            f"{COLORS['dim']}{timestamp}{COLORS['reset']} "
            f"{COLORS['magenta']}[START]{COLORS['reset']} "
            f"{COLORS['dim']}{span.name}{COLORS['reset']}"
        )

    def on_end(self, span: ReadableSpan) -> None:
        if span.attributes and span.attributes.get("__autotraced__"):
            return

        timestamp = datetime.fromtimestamp(span.end_time / 1e9, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]

        span_context = (
            f"{COLORS['dim']}{timestamp}{COLORS['reset']} "
            f"{COLORS['magenta']}[END]{COLORS['reset']} "
            f"{COLORS['dim']}{span.name}{COLORS['reset']}"
        )

        if span.status.status_code == StatusCode.ERROR:
            span_context += f"{COLORS['reset']} {COLORS['red']}[ERROR]{COLORS['reset']}"
        elif span.status.status_code != StatusCode.UNSET:
            span_context += f"{COLORS['reset']} [{span.status.status_code}]"

        duration_ms = (span.end_time - span.start_time) / 1e6
        span_context += f"{COLORS['reset']} ({duration_ms:.2f}ms)"

        print(span_context)

        if self.print_attributes and span.attributes:
            for key, value in span.attributes.items():
                if key.startswith("__"):
                    continue
                str_value = str(value)
                if len(str_value) > 1000:
                    str_value = str_value[:997] + "..."
                print(f"    {COLORS['dim']}{key}: {str_value}{COLORS['reset']}")

        for event in span.events:
            event_time = datetime.fromtimestamp(event.timestamp / 1e9, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]

            severity = event.attributes.get("severity", "info")
            message = event.attributes.get("message", event.name)
            if isinstance(message, (dict, list)):
                message = json.dumps(message, indent=2)

            severity_colors = {
                "error": f"{COLORS['bold']}{COLORS['red']}",
                "warn": f"{COLORS['bold']}{COLORS['yellow']}",
                "info": COLORS["white"],
                "debug": COLORS["dim"],
            }
            msg_color = severity_colors.get(severity, COLORS["white"])

            print(f" {event_time} {msg_color}[{severity.upper()}] {message}{COLORS['reset']}")

            if event.attributes:
                for key, value in event.attributes.items():
                    if key.startswith("__") or key in ["message", "severity"]:
                        continue
                    print(f"   {COLORS['dim']}{key}: {value}{COLORS['reset']}")

    def shutdown(self) -> None:
        """Shutdown the processor."""
        pass

    def force_flush(self, timeout_millis: float | None = None) -> bool:
        """Force flush any pending spans."""
        return True
