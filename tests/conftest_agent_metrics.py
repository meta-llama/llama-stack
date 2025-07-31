# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Pytest configuration for agent metrics tests.

This file contains shared fixtures and configuration for testing agent workflow metrics.
"""

import pytest


# Test markers for organizing agent metrics tests
def pytest_configure(config):
    """Configure pytest markers for agent metrics tests"""
    config.addinivalue_line("markers", "agent_metrics: Agent workflow metrics tests")
    config.addinivalue_line("markers", "agent_metrics_unit: Unit tests for agent metrics")
    config.addinivalue_line("markers", "agent_metrics_integration: Integration tests for agent metrics")
    config.addinivalue_line("markers", "agent_metrics_performance: Performance tests for agent metrics")
    config.addinivalue_line("markers", "agent_metrics_slow: Slow-running metric tests")


@pytest.fixture(scope="session")
def agent_metrics_test_config():
    """Global configuration for agent metrics tests"""
    return {
        "max_test_duration": 30.0,  # Maximum test duration in seconds
        "performance_threshold": 0.01,  # 1% performance overhead threshold
        "metric_batch_size": 100,  # Default batch size for performance tests
        "concurrent_tasks": 10,  # Default concurrent task count
    }


# Cleanup fixture to ensure tests don't interfere with each other
@pytest.fixture(autouse=True)
def cleanup_agent_metrics():
    """Ensure clean state between agent metrics tests"""
    # Pre-test setup
    yield

    # Post-test cleanup
    # Clear any global state that might affect other tests
    from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

    # Reset telemetry global storage
    if "histograms" in _GLOBAL_STORAGE:
        _GLOBAL_STORAGE["histograms"].clear()
    if "counters" in _GLOBAL_STORAGE:
        _GLOBAL_STORAGE["counters"].clear()
    if "up_down_counters" in _GLOBAL_STORAGE:
        _GLOBAL_STORAGE["up_down_counters"].clear()


@pytest.fixture
def fast_mock_config():
    """Configuration for fast mock objects in tests"""
    return {
        "mock_delays": False,  # Disable artificial delays in mocks
        "fast_mode": True,  # Enable fast mode for mocks
        "batch_async": True,  # Batch async operations where possible
    }
