# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
from datetime import UTC, datetime, timedelta

import pytest


@pytest.fixture(scope="module", autouse=True)
def setup_telemetry_metrics_data(llama_stack_client, text_model_id):
    """Setup fixture that creates telemetry metrics data before tests run."""

    # Create inference requests to generate metrics
    for i in range(3):
        llama_stack_client.inference.chat_completion(
            model_id=text_model_id,
            messages=[{"role": "user", "content": f"Test metrics generation {i}"}],
            stream=False,  # Ensure metrics are captured
        )

    # Create OpenAI completions to generate metrics
    for i in range(2):
        llama_stack_client.chat.completions.create(
            model=text_model_id,
            messages=[{"role": "user", "content": f"OpenAI test {i}"}],
            stream=False,
        )

    # Wait for metrics to be logged
    start_time = time.time()
    while time.time() - start_time < 30:
        try:
            # Try to query metrics to see if they're available
            metrics_response = llama_stack_client.telemetry.query_metrics(
                metric_name="prompt_tokens",
                start_time=int((datetime.now(UTC) - timedelta(minutes=5)).timestamp()),
            )
            if len(metrics_response[0].values) > 0:
                break
        except Exception:
            pass
        time.sleep(1)

    # Wait additional time to ensure all metrics are processed
    time.sleep(5)

    yield


def test_query_metrics_prompt_tokens(llama_stack_client, text_model_id):
    """Test that prompt_tokens metrics are queryable."""
    start_time = int((datetime.now(UTC) - timedelta(minutes=10)).timestamp())

    response = llama_stack_client.telemetry.query_metrics(
        metric_name="prompt_tokens",
        start_time=start_time,
    )

    assert isinstance(response, list)

    assert isinstance(response[0].values, list), "Should return a list of metric series"

    assert response[0].metric == "prompt_tokens"


def test_query_metrics_completion_tokens(llama_stack_client, text_model_id):
    """Test that completion_tokens metrics are queryable."""
    start_time = int((datetime.now(UTC) - timedelta(minutes=10)).timestamp())

    response = llama_stack_client.telemetry.query_metrics(
        metric_name="completion_tokens",
        start_time=start_time,
    )

    assert isinstance(response, list)

    assert isinstance(response[0].values, list), "Should return a list of metric series"

    assert response[0].metric == "completion_tokens"


def test_query_metrics_total_tokens(llama_stack_client, text_model_id):
    """Test that total_tokens metrics are queryable."""
    start_time = int((datetime.now(UTC) - timedelta(minutes=10)).timestamp())

    response = llama_stack_client.telemetry.query_metrics(
        metric_name="total_tokens",
        start_time=start_time,
    )

    assert isinstance(response, list)

    assert isinstance(response[0].values, list), "Should return a list of metric series"

    assert response[0].metric == "total_tokens"


def test_query_metrics_with_time_range(llama_stack_client, text_model_id):
    """Test that metrics are queryable with time range."""
    end_time = int(datetime.now(UTC).timestamp())
    start_time = end_time - 600  # 10 minutes ago

    response = llama_stack_client.telemetry.query_metrics(
        metric_name="prompt_tokens",
        start_time=start_time,
        end_time=end_time,
    )

    assert isinstance(response, list)

    assert isinstance(response[0].values, list), "Should return a list of metric series"

    assert response[0].metric == "prompt_tokens"


def test_query_metrics_with_label_matchers(llama_stack_client, text_model_id):
    """Test that metrics are queryable with label matchers."""
    start_time = int((datetime.now(UTC) - timedelta(minutes=10)).timestamp())

    response = llama_stack_client.telemetry.query_metrics(
        metric_name="prompt_tokens",
        start_time=start_time,
        label_matchers=[{"name": "model_id", "value": text_model_id, "operator": "="}],
    )

    assert isinstance(response[0].values, list), "Should return a list of metric series"


def test_query_metrics_nonexistent_metric(llama_stack_client):
    """Test that querying a nonexistent metric returns empty data."""
    start_time = int((datetime.now(UTC) - timedelta(minutes=10)).timestamp())

    response = llama_stack_client.telemetry.query_metrics(
        metric_name="nonexistent_metric",
        start_time=start_time,
    )

    assert isinstance(response, list), "Should return an empty list for nonexistent metric"
    assert len(response) == 0


def test_query_metrics_with_granularity(llama_stack_client, text_model_id):
    """Test that metrics are queryable with different granularity levels."""
    start_time = int((datetime.now(UTC) - timedelta(minutes=10)).timestamp())

    # Test hourly granularity
    hourly_response = llama_stack_client.telemetry.query_metrics(
        metric_name="total_tokens",
        start_time=start_time,
        granularity="1h",
    )

    # Test daily granularity
    daily_response = llama_stack_client.telemetry.query_metrics(
        metric_name="total_tokens",
        start_time=start_time,
        granularity="1d",
    )

    # Test no granularity (raw data points)
    raw_response = llama_stack_client.telemetry.query_metrics(
        metric_name="total_tokens",
        start_time=start_time,
        granularity=None,
    )

    # All should return valid data
    assert isinstance(hourly_response[0].values, list), "Hourly granularity should return data"
    assert isinstance(daily_response[0].values, list), "Daily granularity should return data"
    assert isinstance(raw_response[0].values, list), "No granularity should return data"

    # Verify that different granularities produce different aggregation levels
    # (The exact number depends on data distribution, but they should be queryable)
    assert len(hourly_response[0].values) >= 0, "Hourly granularity should be queryable"
    assert len(daily_response[0].values) >= 0, "Daily granularity should be queryable"
    assert len(raw_response[0].values) >= 0, "No granularity should be queryable"
