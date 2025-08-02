# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from llama_stack.apis.agents import AgentConfig
from llama_stack.apis.inference import SamplingParams
from llama_stack.apis.telemetry import MetricEvent, Telemetry
from llama_stack.providers.inline.agents.meta_reference.agent_instance import (
    TOOL_NAME_MAPPING,
    ChatAgent,
)
from llama_stack.providers.utils.kvstore import InmemoryKVStoreImpl


class TestAgentMetrics:
    """Unit tests for agent workflow metrics"""

    @pytest.fixture
    def mock_telemetry_api(self):
        """Mock telemetry API for unit testing"""
        mock_telemetry = AsyncMock(spec=Telemetry)
        mock_telemetry.log_event = AsyncMock()
        return mock_telemetry

    @pytest.fixture
    def mock_span(self):
        """Mock OpenTelemetry span"""
        mock_span = Mock()
        mock_context = Mock()
        mock_context.trace_id = 0x123456789ABCDEF0  # Non-zero trace_id
        mock_context.span_id = 0xDEADBEEF
        mock_span.get_span_context.return_value = mock_context
        return mock_span

    @pytest.fixture
    def agent_config(self):
        """Basic agent config for testing"""
        return AgentConfig(
            model="test-model",
            instructions="Test agent",
            sampling_params=SamplingParams(),
            tools=[],
            tool_groups=[],
            tool_prompt_format="json",
            input_shields=[],
            output_shields=[],
            max_infer_iters=5,
        )

    @pytest.fixture
    def chat_agent(self, agent_config, mock_telemetry_api):
        """ChatAgent instance with mocked dependencies"""
        # Mock all required dependencies
        mock_inference_api = AsyncMock()
        mock_safety_api = AsyncMock()
        mock_vector_io_api = AsyncMock()
        mock_tool_runtime_api = AsyncMock()
        mock_tool_groups_api = AsyncMock()
        mock_persistence_store = InmemoryKVStoreImpl()

        agent = ChatAgent(
            agent_id="test-agent-123",
            agent_config=agent_config,
            inference_api=mock_inference_api,
            safety_api=mock_safety_api,
            vector_io_api=mock_vector_io_api,
            tool_runtime_api=mock_tool_runtime_api,
            tool_groups_api=mock_tool_groups_api,
            telemetry_api=mock_telemetry_api,
            persistence_store=mock_persistence_store,
            created_at="2024-01-01T00:00:00Z",
            policy=[],
        )
        return agent

    @pytest.fixture
    def chat_agent_no_telemetry(self, agent_config):
        """ChatAgent instance without telemetry for testing graceful degradation"""
        # Mock all required dependencies
        mock_inference_api = AsyncMock()
        mock_safety_api = AsyncMock()
        mock_vector_io_api = AsyncMock()
        mock_tool_runtime_api = AsyncMock()
        mock_tool_groups_api = AsyncMock()
        mock_persistence_store = InmemoryKVStoreImpl()

        agent = ChatAgent(
            agent_id="test-agent-no-telemetry",
            agent_config=agent_config,
            inference_api=mock_inference_api,
            safety_api=mock_safety_api,
            vector_io_api=mock_vector_io_api,
            tool_runtime_api=mock_tool_runtime_api,
            tool_groups_api=mock_tool_groups_api,
            telemetry_api=None,  # No telemetry
            persistence_store=mock_persistence_store,
            created_at="2024-01-01T00:00:00Z",
            policy=[],
        )
        return agent

    def test_construct_agent_metric_success(self, chat_agent, mock_span):
        """Test successful metric construction"""
        with patch(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", return_value=mock_span
        ):
            with patch("time.time", return_value=1234567890.0):
                metric = chat_agent._construct_agent_metric(
                    metric_name="test_metric",
                    value=42,
                    unit="count",
                    labels={"status": "success"},
                )

                assert metric is not None
                assert metric.metric == "test_metric"
                assert metric.value == 42
                assert metric.unit == "count"
                assert metric.timestamp == datetime.fromtimestamp(1234567890.0, tz=UTC)
                assert metric.trace_id == "123456789abcdef0"  # hex format
                assert metric.span_id == "deadbeef"  # hex format
                assert metric.attributes["agent_id"] == "test-agent-123"
                assert metric.attributes["status"] == "success"

    def test_construct_agent_metric_no_span(self, chat_agent):
        """Test metric construction when no span is available"""
        with patch(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", return_value=None
        ):
            metric = chat_agent._construct_agent_metric(
                metric_name="test_metric",
                value=42,
                unit="count",
            )
            assert metric is None

    def test_construct_agent_metric_no_labels(self, chat_agent, mock_span):
        """Test metric construction without labels"""
        with patch(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", return_value=mock_span
        ):
            metric = chat_agent._construct_agent_metric(
                metric_name="test_metric",
                value=42,
                unit="count",
            )

            assert metric is not None
            assert "agent_id" in metric.attributes
            assert len(metric.attributes) == 1  # Only agent_id

    def test_log_agent_metric_safe_success(self, chat_agent, mock_telemetry_api):
        """Test safe metric logging with successful telemetry call"""
        mock_metric = Mock(spec=MetricEvent)
        mock_metric.metric = "test_metric"

        chat_agent._log_agent_metric_safe(mock_metric)

        # Allow async task to complete
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.001))

        # Verify telemetry was called
        mock_telemetry_api.log_event.assert_called_once_with(mock_metric)

    def test_log_agent_metric_safe_no_metric(self, chat_agent, mock_telemetry_api):
        """Test safe metric logging with None metric"""
        chat_agent._log_agent_metric_safe(None)

        # Verify telemetry was not called
        mock_telemetry_api.log_event.assert_not_called()

    def test_log_agent_metric_safe_no_telemetry(self, chat_agent_no_telemetry):
        """Test safe metric logging without telemetry API"""
        mock_metric = Mock(spec=MetricEvent)

        # Should not raise exception
        chat_agent_no_telemetry._log_agent_metric_safe(mock_metric)

    def test_log_agent_metric_safe_telemetry_exception(self, chat_agent, mock_telemetry_api):
        """Test safe metric logging when telemetry raises exception"""
        mock_metric = Mock(spec=MetricEvent)
        mock_metric.metric = "test_metric"
        mock_telemetry_api.log_event.side_effect = Exception("Telemetry error")

        # Should not raise exception, just log warning
        with patch("llama_stack.providers.inline.agents.meta_reference.agent_instance.logger"):
            chat_agent._log_agent_metric_safe(mock_metric)
            # Allow async task to complete and fail
            asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.001))

    def test_log_workflow_completion_success(self, chat_agent, mock_telemetry_api, mock_span):
        """Test workflow completion logging for successful workflow"""
        with patch(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", return_value=mock_span
        ):
            with patch("time.time", return_value=1234567890.0):
                chat_agent._log_workflow_completion("completed", 15.5)

        # Allow async tasks to complete
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.001))

        # Should have called telemetry twice (completion + duration)
        assert mock_telemetry_api.log_event.call_count == 2

        # Verify the metrics
        calls = mock_telemetry_api.log_event.call_args_list
        completion_metric = calls[0][0][0]
        duration_metric = calls[1][0][0]

        # Check completion metric
        assert completion_metric.metric == "llama_stack_agent_workflows_total"
        assert completion_metric.value == 1
        assert completion_metric.attributes["status"] == "completed"

        # Check duration metric
        assert duration_metric.metric == "llama_stack_agent_workflow_duration_seconds"
        assert duration_metric.value == 15.5
        assert duration_metric.unit == "s"

    def test_log_workflow_completion_failed(self, chat_agent, mock_telemetry_api, mock_span):
        """Test workflow completion logging for failed workflow"""
        with patch(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", return_value=mock_span
        ):
            with patch("time.time", return_value=1234567890.0):
                chat_agent._log_workflow_completion("failed", 3.2)

        # Allow async tasks to complete
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.001))

        # Verify failure status
        calls = mock_telemetry_api.log_event.call_args_list
        completion_metric = calls[0][0][0]
        assert completion_metric.attributes["status"] == "failed"

    def test_log_step_execution(self, chat_agent, mock_telemetry_api, mock_span):
        """Test step execution logging"""
        with patch(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", return_value=mock_span
        ):
            with patch("time.time", return_value=1234567890.0):
                chat_agent._log_step_execution(3)

        # Allow async task to complete
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.001))

        mock_telemetry_api.log_event.assert_called_once()

        metric = mock_telemetry_api.log_event.call_args[0][0]
        assert metric.metric == "llama_stack_agent_steps_total"
        assert metric.value == 3
        assert metric.unit == "1"

    def test_log_tool_usage_with_mapping(self, chat_agent, mock_telemetry_api, mock_span):
        """Test tool usage logging with name mapping"""
        with patch(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", return_value=mock_span
        ):
            with patch("time.time", return_value=1234567890.0):
                # Test mapping brave_search -> web_search
                chat_agent._log_tool_usage("brave_search")

        # Allow async task to complete
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.001))

        mock_telemetry_api.log_event.assert_called_once()

        metric = mock_telemetry_api.log_event.call_args[0][0]
        assert metric.metric == "llama_stack_agent_tool_calls_total"
        assert metric.value == 1
        assert metric.attributes["tool"] == "web_search"  # Should be mapped

    def test_log_tool_usage_no_mapping(self, chat_agent, mock_telemetry_api, mock_span):
        """Test tool usage logging without name mapping"""
        with patch(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", return_value=mock_span
        ):
            with patch("time.time", return_value=1234567890.0):
                # Test unmapped tool name
                chat_agent._log_tool_usage("custom_tool")

        # Allow async task to complete
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.001))

        metric = mock_telemetry_api.log_event.call_args[0][0]
        assert metric.attributes["tool"] == "custom_tool"  # Should remain unchanged

    def test_tool_name_mapping_constants(self):
        """Test tool name mapping constants"""
        assert TOOL_NAME_MAPPING["knowledge_search"] == "rag"
        assert TOOL_NAME_MAPPING["web_search"] == "web_search"
        assert TOOL_NAME_MAPPING["brave_search"] == "web_search"

    def test_all_metric_methods_handle_no_telemetry(self, chat_agent_no_telemetry):
        """Test that all metric methods handle missing telemetry gracefully"""
        # These should all complete without errors
        chat_agent_no_telemetry._log_workflow_completion("completed", 1.0)
        chat_agent_no_telemetry._log_step_execution(1)
        chat_agent_no_telemetry._log_tool_usage("test_tool")

    def test_metric_construction_with_float_values(self, chat_agent, mock_span):
        """Test metric construction with float values (for duration)"""
        with patch(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", return_value=mock_span
        ):
            metric = chat_agent._construct_agent_metric(
                metric_name="duration_metric",
                value=15.7,
                unit="s",
            )

            assert metric.value == 15.7
            assert isinstance(metric.value, float)

    def test_metric_construction_with_int_values(self, chat_agent, mock_span):
        """Test metric construction with int values (for counters)"""
        with patch(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", return_value=mock_span
        ):
            metric = chat_agent._construct_agent_metric(
                metric_name="counter_metric",
                value=42,
                unit="count",
            )

            assert metric.value == 42
            assert isinstance(metric.value, int)

    async def test_concurrent_metric_logging(self, chat_agent, mock_telemetry_api, mock_span):
        """Test that concurrent metric logging doesn't interfere"""
        with patch(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", return_value=mock_span
        ):
            with patch("time.time", return_value=1234567890.0):
                # Log multiple metrics concurrently
                for i in range(10):
                    chat_agent._log_step_execution(i)
                    chat_agent._log_tool_usage(f"tool_{i}")

        # Allow all async tasks to complete
        await asyncio.sleep(0.01)

        # Should have 20 total calls (10 step + 10 tool)
        assert mock_telemetry_api.log_event.call_count == 20

    def test_agent_id_consistency(self, chat_agent, mock_telemetry_api, mock_span):
        """Test that agent_id is consistently included in all metrics"""
        with patch(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", return_value=mock_span
        ):
            # Log various metrics
            chat_agent._log_workflow_completion("completed", 1.0)
            chat_agent._log_step_execution(1)
            chat_agent._log_tool_usage("test_tool")

        # Allow async tasks to complete
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.001))

        # Check all metrics have the correct agent_id
        calls = mock_telemetry_api.log_event.call_args_list
        for call in calls:
            metric = call[0][0]
            assert metric.attributes["agent_id"] == "test-agent-123"

    def test_metric_units_are_correct(self, chat_agent, mock_telemetry_api, mock_span):
        """Test that metric units are set correctly"""
        with patch(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", return_value=mock_span
        ):
            chat_agent._log_workflow_completion("completed", 1.0)
            chat_agent._log_step_execution(1)
            chat_agent._log_tool_usage("test_tool")

        # Allow async tasks to complete
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.001))

        calls = mock_telemetry_api.log_event.call_args_list

        # Workflow total should be "1" (dimensionless)
        completion_metric = calls[0][0][0]
        assert completion_metric.unit == "1"

        # Duration should be "s" (seconds)
        duration_metric = calls[1][0][0]
        assert duration_metric.unit == "s"

        # Steps should be "1" (dimensionless)
        steps_metric = calls[2][0][0]
        assert steps_metric.unit == "1"

        # Tool calls should be "1" (dimensionless)
        tool_metric = calls[3][0][0]
        assert tool_metric.unit == "1"
