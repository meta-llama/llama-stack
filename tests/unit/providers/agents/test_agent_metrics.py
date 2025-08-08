# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from opentelemetry.trace import SpanContext, TraceFlags

from llama_stack.providers.inline.agents.meta_reference.agent_instance import ChatAgent


class FakeSpan:
    def __init__(self, trace_id: int = 123, span_id: int = 456):
        self._context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=False,
            trace_flags=TraceFlags(0x01),
        )

    def get_span_context(self):
        return self._context


@pytest.fixture
def agent():
    return ChatAgent(
        agent_id="test-agent",
        agent_config=Mock(),
        inference_api=Mock(),
        safety_api=Mock(),
        tool_runtime_api=Mock(),
        tool_groups_api=Mock(),
        vector_io_api=Mock(),
        telemetry_api=AsyncMock(),
        persistence_store=Mock(),
        created_at="2025-01-01T00:00:00Z",
        policy=[],
    )


class TestAgentMetrics:
    def setup_method(self):
        self.captured_metrics = []

    async def _setup_mocks(self, agent, monkeypatch, trace_id=123):
        fake_span = FakeSpan(trace_id=trace_id)
        monkeypatch.setattr(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", lambda: fake_span
        )

        async def capture_metric(metric):
            self.captured_metrics.append(metric)

        agent.telemetry_api.log_event = capture_metric

        pending_tasks = []

        def mock_create_task(coro, **kwargs):
            pending_tasks.append(coro)
            return AsyncMock()

        monkeypatch.setattr(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.asyncio.create_task", mock_create_task
        )
        return pending_tasks

    async def test_step_metrics(self, agent, monkeypatch):
        pending_tasks = await self._setup_mocks(agent, monkeypatch)

        agent._log_step_execution()
        await asyncio.gather(*pending_tasks)

        assert len(self.captured_metrics) == 1
        metric = self.captured_metrics[0]
        assert metric.metric == "llama_stack_agent_steps_total"
        assert metric.value == 1
        assert metric.attributes["agent_id"] == "test-agent"

    async def test_workflow_metrics(self, agent, monkeypatch):
        pending_tasks = await self._setup_mocks(agent, monkeypatch)

        agent._log_workflow_completion("completed", 2.5)
        await asyncio.gather(*pending_tasks)

        assert len(self.captured_metrics) == 2
        assert self.captured_metrics[0].metric == "llama_stack_agent_workflows_total"
        assert self.captured_metrics[0].attributes["status"] == "completed"
        assert self.captured_metrics[1].metric == "llama_stack_agent_workflow_duration_seconds"
        assert self.captured_metrics[1].value == 2.5

    async def test_tool_metrics(self, agent, monkeypatch):
        pending_tasks = await self._setup_mocks(agent, monkeypatch)

        agent._log_tool_usage("web_search")
        agent._log_tool_usage("knowledge_search")
        await asyncio.gather(*pending_tasks)

        assert len(self.captured_metrics) == 2
        assert self.captured_metrics[0].attributes["tool"] == "web_search"
        assert self.captured_metrics[1].attributes["tool"] == "rag"

    def test_no_telemetry(self):
        agent = ChatAgent(
            agent_id="test",
            agent_config=Mock(),
            inference_api=Mock(),
            safety_api=Mock(),
            tool_runtime_api=Mock(),
            tool_groups_api=Mock(),
            vector_io_api=Mock(),
            telemetry_api=None,
            persistence_store=Mock(),
            created_at="2025-01-01T00:00:00Z",
            policy=[],
        )

        agent._log_step_execution()
        agent._log_workflow_completion("failed", 1.0)

    def test_no_span(self, agent, monkeypatch):
        monkeypatch.setattr(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", lambda: None
        )

        agent._log_step_execution()
        agent.telemetry_api.log_event.assert_not_called()

    def test_invalid_trace_id(self, agent, monkeypatch):
        fake_span = FakeSpan(trace_id=0)
        monkeypatch.setattr(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", lambda: fake_span
        )

        agent._log_step_execution()
        agent.telemetry_api.log_event.assert_not_called()
