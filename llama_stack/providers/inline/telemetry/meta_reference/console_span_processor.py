# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanProcessor

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
    """A SpanProcessor that prints spans to the console with color formatting."""

    def on_start(self, span: ReadableSpan, parent_context=None) -> None:
        """Called when a span starts."""
        timestamp = datetime.utcfromtimestamp(span.start_time / 1e9).strftime(
            "%H:%M:%S.%f"
        )[:-3]

        print(
            f"{COLORS['dim']}{timestamp}{COLORS['reset']} "
            f"{COLORS['magenta']}[START]{COLORS['reset']} "
            f"{COLORS['cyan']}{span.name}{COLORS['reset']}"
        )

    def on_end(self, span: ReadableSpan) -> None:
        """Called when a span ends."""
        timestamp = datetime.utcfromtimestamp(span.end_time / 1e9).strftime(
            "%H:%M:%S.%f"
        )[:-3]

        # Build the span context string
        span_context = (
            f"{COLORS['dim']}{timestamp}{COLORS['reset']} "
            f"{COLORS['magenta']}[END]{COLORS['reset']} "
            f"{COLORS['cyan']}{span.name}{COLORS['reset']} "
        )

        # Add status if not OK
        if span.status.status_code != 0:  # UNSET or ERROR
            status_color = (
                COLORS["red"] if span.status.status_code == 2 else COLORS["yellow"]
            )
            span_context += (
                f" {status_color}[{span.status.status_code}]{COLORS['reset']}"
            )

        # Add duration
        duration_ms = (span.end_time - span.start_time) / 1e6
        span_context += f" {COLORS['dim']}({duration_ms:.2f}ms){COLORS['reset']}"

        # Print the main span line
        print(span_context)

        # Print attributes indented
        if span.attributes:
            for key, value in span.attributes.items():
                # Skip internal attributes; also rename these internal attributes to have underscores
                if key in ("class", "method", "type", "__root__", "__ttl__"):
                    continue
                print(f"  {COLORS['dim']}{key}: {value}{COLORS['reset']}")

        # Print events indented
        for event in span.events:
            event_time = datetime.utcfromtimestamp(event.timestamp / 1e9).strftime(
                "%H:%M:%S.%f"
            )[:-3]
            print(
                f"  {COLORS['dim']}{event_time}{COLORS['reset']} "
                f"{COLORS['cyan']}[EVENT]{COLORS['reset']} {event.name}"
            )
            if event.attributes:
                for key, value in event.attributes.items():
                    print(f"    {COLORS['dim']}{key}: {value}{COLORS['reset']}")

    def shutdown(self) -> None:
        """Shutdown the processor."""
        pass

    def force_flush(self, timeout_millis: float = None) -> bool:
        """Force flush any pending spans."""
        return True
