# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from unittest.mock import AsyncMock, Mock, patch

from llama_stack.providers.inline.agents.meta_reference.agent_instance import ChatAgent


class TestAgentMetricsIntegration:
    """Smoke test for agent metrics integration"""

    async def test_agent_metrics_methods_exist_and_work(self):
        """Test that metrics methods exist and can be called without errors"""
        # Create a minimal agent instance with mocked dependencies
        telemetry_api = AsyncMock()
        telemetry_api.logged_events = []

        async def mock_log_event(event):
            telemetry_api.logged_events.append(event)

        telemetry_api.log_event = mock_log_event

        agent = ChatAgent(
            agent_id="test-agent",
            agent_config=Mock(),
            inference_api=Mock(),
            safety_api=Mock(),
            tool_runtime_api=Mock(),
            tool_groups_api=Mock(),
            vector_io_api=Mock(),
            telemetry_api=telemetry_api,
            persistence_store=Mock(),
            created_at="2025-01-01T00:00:00Z",
            policy=[],
        )

        with patch("llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span") as mock_span:
            mock_span.return_value = Mock(get_span_context=Mock(return_value=Mock(trace_id=123, span_id=456)))

            # Test all metrics methods work
            agent._track_step()
            agent._track_workflow("completed", 2.5)
            agent._track_tool("web_search")

            # Wait for async operations
            await asyncio.sleep(0.01)

            # Basic verification that telemetry was called
            assert len(telemetry_api.logged_events) >= 3

            # Verify we can call the methods without exceptions
            agent._track_tool("knowledge_search")  # Test tool mapping
            await asyncio.sleep(0.01)

            assert len(telemetry_api.logged_events) >= 4
