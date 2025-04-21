# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

import pytest

from llama_stack.distribution.server.server import create_sse_event, sse_generator


@pytest.mark.asyncio
async def test_sse_generator_basic():
    # An AsyncIterator wrapped in an Awaitable, just like our web methods
    async def async_event_gen():
        async def event_gen():
            yield "Test event 1"
            yield "Test event 2"

        return event_gen()

    sse_gen = sse_generator(async_event_gen())
    assert sse_gen is not None

    # Test that the events are streamed correctly
    seen_events = []
    async for event in sse_gen:
        seen_events.append(event)
    assert len(seen_events) == 2
    assert seen_events[0] == create_sse_event("Test event 1")
    assert seen_events[1] == create_sse_event("Test event 2")


@pytest.mark.asyncio
async def test_sse_generator_client_disconnected():
    # An AsyncIterator wrapped in an Awaitable, just like our web methods
    async def async_event_gen():
        async def event_gen():
            yield "Test event 1"
            # Simulate a client disconnect before emitting event 2
            raise asyncio.CancelledError()

        return event_gen()

    sse_gen = sse_generator(async_event_gen())
    assert sse_gen is not None

    # Start reading the events, ensuring this doesn't raise an exception
    seen_events = []
    async for event in sse_gen:
        seen_events.append(event)
    assert len(seen_events) == 1
    assert seen_events[0] == create_sse_event("Test event 1")
