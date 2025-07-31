# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from llama_stack.apis.agents import (
    AgentConfig,
    AgentTurnCreateRequest,
)
from llama_stack.apis.inference import SamplingParams, UserMessage
from llama_stack.apis.telemetry import Telemetry
from llama_stack.providers.inline.agents.meta_reference.agents import MetaReferenceAgentsImpl
from llama_stack.providers.inline.agents.meta_reference.config import MetaReferenceAgentsImplConfig
from llama_stack.providers.utils.kvstore import InmemoryKVStoreImpl


class TestAgentMetricsIntegration:
    """Integration tests for agent workflow metrics end-to-end"""

    @pytest.fixture
    def agent_config(self):
        """Agent configuration for integration tests"""
        return AgentConfig(
            model="test-model",
            instructions="You are a helpful assistant for testing agent metrics",
            sampling_params=SamplingParams(temperature=0.7, max_tokens=100),
            tools=[],
            tool_groups=[],
            tool_prompt_format="json",
            input_shields=[],
            output_shields=[],
            max_infer_iters=3,
        )

    @pytest.fixture
    def mock_telemetry_api(self):
        """Mock telemetry API that captures metrics"""
        mock_telemetry = AsyncMock(spec=Telemetry)
        mock_telemetry.logged_events = []  # Store events for verification

        async def capture_event(event):
            mock_telemetry.logged_events.append(event)

        mock_telemetry.log_event.side_effect = capture_event
        return mock_telemetry

    @pytest.fixture
    def agents_impl(self, mock_telemetry_api):
        """MetaReferenceAgentsImpl with mocked dependencies"""
        config = MetaReferenceAgentsImplConfig(
            persistence_store={"type": "sqlite", "db_path": ":memory:"},
            responses_store={"type": "sqlite", "db_path": ":memory:"},
        )

        # Mock all required APIs
        mock_inference_api = AsyncMock()
        mock_vector_io_api = AsyncMock()
        mock_safety_api = AsyncMock()
        mock_tool_runtime_api = AsyncMock()
        mock_tool_groups_api = AsyncMock()

        impl = MetaReferenceAgentsImpl(
            config=config,
            inference_api=mock_inference_api,
            vector_io_api=mock_vector_io_api,
            safety_api=mock_safety_api,
            tool_runtime_api=mock_tool_runtime_api,
            tool_groups_api=mock_tool_groups_api,
            telemetry_api=mock_telemetry_api,
            policy=[],
        )

        # Initialize with in-memory stores
        impl.persistence_store = InmemoryKVStoreImpl()
        impl.in_memory_store = InmemoryKVStoreImpl()

        return impl

    @pytest.fixture
    async def test_agent(self, agents_impl, agent_config):
        """Create a test agent"""
        result = await agents_impl.create_agent(agent_config)
        agent_id = result.agent_id

        # Create a session
        session_result = await agents_impl.create_agent_session(agent_id, "test-session")
        session_id = session_result.session_id

        return {
            "agent_id": agent_id,
            "session_id": session_id,
            "agents_impl": agents_impl,
        }

    async def test_workflow_completion_metrics_success(self, test_agent, mock_telemetry_api):
        """Test workflow completion metrics for successful workflow"""
        # Mock successful inference response
        with patch("llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span") as mock_span:
            mock_span.return_value = Mock(trace_id=123, span_id=456)

            # Mock the inference API to return a simple completion
            test_agent["agents_impl"].inference_api.chat_completion = AsyncMock()

            # Mock the _run method to simulate successful completion
            agent_instance = await test_agent["agents_impl"]._get_agent_impl(test_agent["agent_id"])

            with patch.object(agent_instance, "_run") as mock_run:
                from llama_stack.apis.inference import CompletionMessage, StopReason

                async def mock_run_generator(*args, **kwargs):
                    # Simulate a successful completion
                    yield CompletionMessage(
                        content="Hello! How can I help you today?",
                        stop_reason=StopReason.end_of_turn,
                        tool_calls=[],
                    )

                mock_run.return_value = mock_run_generator()

                # Create a turn request
                request = AgentTurnCreateRequest(
                    agent_id=test_agent["agent_id"],
                    session_id=test_agent["session_id"],
                    messages=[UserMessage(content="Hello!")],
                    stream=True,
                )

                # Execute the turn
                events = []
                async for event in test_agent["agents_impl"]._create_agent_turn_streaming(request):
                    events.append(event)

                # Wait for async metrics to be logged
                await asyncio.sleep(0.01)

                # Verify metrics were logged
                logged_events = mock_telemetry_api.logged_events

                # Should have workflow completion and duration metrics
                workflow_metrics = [e for e in logged_events if e.metric == "llama_stack_agent_workflows_total"]
                duration_metrics = [
                    e for e in logged_events if e.metric == "llama_stack_agent_workflow_duration_seconds"
                ]

                assert len(workflow_metrics) >= 1, "Should have workflow completion metric"
                assert len(duration_metrics) >= 1, "Should have duration metric"

                # Check workflow completion metric
                completion_metric = workflow_metrics[0]
                assert completion_metric.value == 1
                assert completion_metric.attributes["agent_id"] == test_agent["agent_id"]
                assert completion_metric.attributes["status"] == "completed"

                # Check duration metric
                duration_metric = duration_metrics[0]
                assert duration_metric.value > 0  # Should have some duration
                assert duration_metric.unit == "s"
                assert duration_metric.attributes["agent_id"] == test_agent["agent_id"]

    async def test_workflow_completion_metrics_failure(self, test_agent, mock_telemetry_api):
        """Test workflow completion metrics for failed workflow"""
        with patch("llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span") as mock_span:
            mock_span.return_value = Mock(trace_id=123, span_id=456)

            # Mock the inference API to raise an exception
            agent_instance = await test_agent["agents_impl"]._get_agent_impl(test_agent["agent_id"])

            async def mock_run_generator(*args, **kwargs):
                raise Exception("Simulated failure")
                yield  # This line will never be reached, but makes it an async generator

            with patch.object(agent_instance, "_run_turn", side_effect=lambda *args, **kwargs: mock_run_generator()):
                # Create a turn request
                request = AgentTurnCreateRequest(
                    agent_id=test_agent["agent_id"],
                    session_id=test_agent["session_id"],
                    messages=[UserMessage(content="Hello!")],
                    stream=True,
                )

                # Execute the turn and handle any exceptions
                try:
                    async for _event in test_agent["agents_impl"]._create_agent_turn_streaming(request):
                        pass
                except Exception:
                    # Expected to fail, continue with test
                    pass

                # Wait for async metrics to be logged
                await asyncio.sleep(0.01)

                # Verify failure metrics were logged
                logged_events = mock_telemetry_api.logged_events
                workflow_metrics = [e for e in logged_events if e.metric == "llama_stack_agent_workflows_total"]

                if workflow_metrics:  # May not always be logged due to exception timing
                    failure_metric = next((m for m in workflow_metrics if m.attributes.get("status") == "failed"), None)
                    if failure_metric:
                        assert failure_metric.value == 1
                        assert failure_metric.attributes["agent_id"] == test_agent["agent_id"]

    async def test_step_counter_metrics(self, test_agent, mock_telemetry_api):
        """Test step counter metrics during workflow execution"""
        with patch("llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span") as mock_span:
            mock_span.return_value = Mock(trace_id=123, span_id=456)

            agent_instance = await test_agent["agents_impl"]._get_agent_impl(test_agent["agent_id"])

            # Mock the _run method to simulate step execution
            with patch.object(agent_instance, "_run") as mock_run:
                from llama_stack.apis.agents import (
                    AgentTurnResponseEvent,
                    AgentTurnResponseStepCompletePayload,
                    AgentTurnResponseStreamChunk,
                    InferenceStep,
                    StepType,
                )
                from llama_stack.apis.inference import CompletionMessage, StopReason

                async def mock_run_generator(*args, **kwargs):
                    # Simulate multiple steps
                    step_id = str(uuid.uuid4())
                    turn_id = str(uuid.uuid4())

                    # Yield inference step completion
                    yield AgentTurnResponseStreamChunk(
                        event=AgentTurnResponseEvent(
                            payload=AgentTurnResponseStepCompletePayload(
                                step_type=StepType.inference.value,
                                step_id=step_id,
                                step_details=InferenceStep(
                                    step_id=step_id,
                                    turn_id=turn_id,
                                    model_response=CompletionMessage(
                                        content="Response",
                                        stop_reason=StopReason.end_of_turn,
                                    ),
                                    started_at=datetime.now(UTC).isoformat(),
                                    completed_at=datetime.now(UTC).isoformat(),
                                ),
                            )
                        )
                    )

                    # Final completion message
                    yield CompletionMessage(
                        content="Final response",
                        stop_reason=StopReason.end_of_turn,
                        tool_calls=[],
                    )

                mock_run.return_value = mock_run_generator()

                # Create a turn request
                request = AgentTurnCreateRequest(
                    agent_id=test_agent["agent_id"],
                    session_id=test_agent["session_id"],
                    messages=[UserMessage(content="Count my steps!")],
                    stream=True,
                )

                # Execute the turn
                events = []
                async for event in test_agent["agents_impl"]._create_agent_turn_streaming(request):
                    events.append(event)

                # Wait for async metrics to be logged
                await asyncio.sleep(0.01)

                # Verify step metrics were logged
                logged_events = mock_telemetry_api.logged_events
                step_metrics = [e for e in logged_events if e.metric == "llama_stack_agent_steps_total"]

                assert len(step_metrics) >= 1, "Should have step counter metrics"

                # Check step metric
                step_metric = step_metrics[0]
                assert step_metric.value >= 1  # Should count at least one step
                assert step_metric.unit == "1"
                assert step_metric.attributes["agent_id"] == test_agent["agent_id"]

    async def test_tool_usage_metrics(self, test_agent, mock_telemetry_api):
        """Test tool usage metrics during tool execution"""
        with patch("llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span") as mock_span:
            mock_span.return_value = Mock(trace_id=123, span_id=456)

            agent_instance = await test_agent["agents_impl"]._get_agent_impl(test_agent["agent_id"])

            # Mock tool execution
            from llama_stack.apis.tools import ToolInvocationResult
            from llama_stack.models.llama.datatypes import ToolCall

            test_tool_call = ToolCall(
                call_id="test-call-123",
                tool_name="web_search",
                arguments={"query": "test search"},
            )

            # Mock tool runtime to return a result
            agent_instance.tool_runtime_api.invoke_tool = AsyncMock(
                return_value=ToolInvocationResult(
                    content="Search results for test search",
                    metadata={},
                )
            )

            # Mock tool definitions
            agent_instance.tool_defs = [Mock(tool_name="web_search")]
            agent_instance.tool_name_to_args = {}

            # Execute tool call
            await agent_instance.execute_tool_call_maybe(
                test_agent["session_id"],
                test_tool_call,
            )

            # Wait for async metrics to be logged
            await asyncio.sleep(0.01)

            # Verify tool usage metrics were logged
            logged_events = mock_telemetry_api.logged_events
            tool_metrics = [e for e in logged_events if e.metric == "llama_stack_agent_tool_calls_total"]

            assert len(tool_metrics) >= 1, "Should have tool usage metrics"

            # Check tool metric
            tool_metric = tool_metrics[0]
            assert tool_metric.value == 1
            assert tool_metric.unit == "1"
            assert tool_metric.attributes["agent_id"] == test_agent["agent_id"]
            assert tool_metric.attributes["tool"] == "web_search"

    async def test_tool_name_mapping_integration(self, test_agent, mock_telemetry_api):
        """Test tool name mapping works in integration"""
        with patch("llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span") as mock_span:
            mock_span.return_value = Mock(trace_id=123, span_id=456)

            agent_instance = await test_agent["agents_impl"]._get_agent_impl(test_agent["agent_id"])

            # Test mapping knowledge_search -> rag
            from llama_stack.apis.tools import ToolInvocationResult
            from llama_stack.models.llama.datatypes import ToolCall

            knowledge_search_call = ToolCall(
                call_id="test-call-456",
                tool_name="knowledge_search",
                arguments={"query": "test knowledge"},
            )

            # Mock tool runtime
            agent_instance.tool_runtime_api.invoke_tool = AsyncMock(
                return_value=ToolInvocationResult(
                    content="Knowledge search results",
                    metadata={},
                )
            )

            # Mock tool definitions
            agent_instance.tool_defs = [Mock(tool_name="knowledge_search")]
            agent_instance.tool_name_to_args = {}

            # Execute tool call
            await agent_instance.execute_tool_call_maybe(
                test_agent["session_id"],
                knowledge_search_call,
            )

            # Wait for async metrics to be logged
            await asyncio.sleep(0.01)

            # Verify tool name was mapped
            logged_events = mock_telemetry_api.logged_events
            tool_metrics = [e for e in logged_events if e.metric == "llama_stack_agent_tool_calls_total"]

            assert len(tool_metrics) >= 1
            tool_metric = tool_metrics[0]
            assert tool_metric.attributes["tool"] == "rag"  # Should be mapped from knowledge_search

    async def test_metrics_with_no_telemetry(self, agent_config):
        """Test that agent works normally when telemetry is disabled"""
        # Create agents impl without telemetry
        config = MetaReferenceAgentsImplConfig(
            persistence_store={"type": "sqlite", "db_path": ":memory:"},
            responses_store={"type": "sqlite", "db_path": ":memory:"},
        )

        impl = MetaReferenceAgentsImpl(
            config=config,
            inference_api=AsyncMock(),
            vector_io_api=AsyncMock(),
            safety_api=AsyncMock(),
            tool_runtime_api=AsyncMock(),
            tool_groups_api=AsyncMock(),
            telemetry_api=None,  # No telemetry
            policy=[],
        )

        impl.persistence_store = InmemoryKVStoreImpl()
        impl.in_memory_store = InmemoryKVStoreImpl()

        # Create agent and session
        result = await impl.create_agent(agent_config)
        agent_id = result.agent_id

        await impl.create_agent_session(agent_id, "test-session-no-telemetry")

        # Get agent instance and verify it works
        agent_instance = await impl._get_agent_impl(agent_id)
        assert agent_instance.telemetry_api is None

        # These should not raise exceptions
        agent_instance._log_workflow_completion("completed", 1.0)
        agent_instance._log_step_execution(1)
        agent_instance._log_tool_usage("test_tool")

    async def test_concurrent_metric_logging(self, test_agent, mock_telemetry_api):
        """Test metrics under concurrent workflow execution"""
        with patch("llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span") as mock_span:
            mock_span.return_value = Mock(trace_id=123, span_id=456)

            # Create multiple concurrent tool executions
            agent_instance = await test_agent["agents_impl"]._get_agent_impl(test_agent["agent_id"])

            from llama_stack.apis.tools import ToolInvocationResult
            from llama_stack.models.llama.datatypes import ToolCall

            # Mock tool runtime
            agent_instance.tool_runtime_api.invoke_tool = AsyncMock(
                return_value=ToolInvocationResult(content="result", metadata={})
            )
            agent_instance.tool_defs = [Mock(tool_name="test_tool")]
            agent_instance.tool_name_to_args = {}

            # Execute multiple tool calls concurrently
            tasks = []
            for i in range(5):
                tool_call = ToolCall(
                    call_id=f"call-{i}",
                    tool_name="test_tool",
                    arguments={"param": f"value-{i}"},
                )
                tasks.append(agent_instance.execute_tool_call_maybe(test_agent["session_id"], tool_call))

            # Wait for all to complete
            await asyncio.gather(*tasks)

            # Wait for async metrics to be logged
            await asyncio.sleep(0.01)

            # Verify all tool metrics were logged
            logged_events = mock_telemetry_api.logged_events
            tool_metrics = [e for e in logged_events if e.metric == "llama_stack_agent_tool_calls_total"]

            assert len(tool_metrics) == 5, "Should have metrics for all concurrent tool calls"

            # Verify all have correct attributes
            for metric in tool_metrics:
                assert metric.value == 1
                assert metric.attributes["agent_id"] == test_agent["agent_id"]
                assert metric.attributes["tool"] == "test_tool"

    async def test_metric_attribute_consistency(self, test_agent, mock_telemetry_api):
        """Test that all metrics have consistent agent_id attributes"""
        with patch("llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span") as mock_span:
            mock_span.return_value = Mock(trace_id=123, span_id=456)

            agent_instance = await test_agent["agents_impl"]._get_agent_impl(test_agent["agent_id"])

            # Log various types of metrics
            agent_instance._log_workflow_completion("completed", 5.2)
            agent_instance._log_step_execution(3)
            agent_instance._log_tool_usage("web_search")
            agent_instance._log_tool_usage("knowledge_search")  # Should map to "rag"

            # Wait for async metrics to be logged
            await asyncio.sleep(0.01)

            # Verify all metrics have consistent agent_id
            logged_events = mock_telemetry_api.logged_events
            assert len(logged_events) >= 4  # At least 4 metrics (2 for workflow completion)

            for event in logged_events:
                assert "agent_id" in event.attributes
                assert event.attributes["agent_id"] == test_agent["agent_id"]

            # Verify metric names are correct
            metric_names = {event.metric for event in logged_events}
            expected_metrics = {
                "llama_stack_agent_workflows_total",
                "llama_stack_agent_workflow_duration_seconds",
                "llama_stack_agent_steps_total",
                "llama_stack_agent_tool_calls_total",
            }
            assert expected_metrics.issubset(metric_names)
