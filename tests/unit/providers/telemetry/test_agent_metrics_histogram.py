# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import Mock

import pytest

from llama_stack.apis.telemetry import MetricEvent, MetricType
from llama_stack.providers.inline.telemetry.meta_reference.config import TelemetryConfig
from llama_stack.providers.inline.telemetry.meta_reference.telemetry import TelemetryAdapter


@pytest.fixture
def adapter():
    config = TelemetryConfig(service_name="test-service", sinks=[])
    adapter = TelemetryAdapter(config, {})
    adapter.meter = Mock()
    return adapter


class TestHistogramMetrics:
    def setup_method(self):
        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        _GLOBAL_STORAGE.clear()
        _GLOBAL_STORAGE["histograms"] = {}
        _GLOBAL_STORAGE["counters"] = {}

    def test_create_histogram(self, adapter):
        mock_histogram = Mock()
        adapter.meter.create_histogram.return_value = mock_histogram

        result = adapter._get_or_create_histogram("test_histogram", "s")

        assert result == mock_histogram
        adapter.meter.create_histogram.assert_called_once_with(
            name="test_histogram", unit="s", description="Histogram for test_histogram"
        )

    def test_reuse_existing_histogram(self, adapter):
        mock_histogram = Mock()
        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        _GLOBAL_STORAGE["histograms"]["existing"] = mock_histogram

        result = adapter._get_or_create_histogram("existing", "ms")

        assert result == mock_histogram
        adapter.meter.create_histogram.assert_not_called()

    def test_duration_metrics_use_histogram(self, adapter):
        mock_histogram = Mock()
        adapter.meter.create_histogram.return_value = mock_histogram

        metric = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="workflow_duration_seconds",
            value=15.7,
            timestamp=1234567890.0,
            unit="s",
            attributes={"agent_id": "test"},
            metric_type=MetricType.HISTOGRAM,
        )

        adapter._log_metric(metric)

        adapter.meter.create_histogram.assert_called_once()
        mock_histogram.record.assert_called_once_with(15.7, attributes={"agent_id": "test"})

    def test_counter_metrics_use_counter(self, adapter):
        mock_counter = Mock()
        adapter.meter.create_counter.return_value = mock_counter

        metric = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="workflows_total",
            value=1,
            timestamp=1234567890.0,
            unit="1",
            attributes={"agent_id": "test", "status": "completed"},
        )

        adapter._log_metric(metric)

        adapter.meter.create_counter.assert_called_once()
        adapter.meter.create_histogram.assert_not_called()
        mock_counter.add.assert_called_once_with(1, attributes={"agent_id": "test", "status": "completed"})

    def test_no_meter_graceful(self, adapter):
        adapter.meter = None
        metric = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="test_duration_seconds",
            value=1.0,
            timestamp=1234567890.0,
            unit="s",
            attributes={},
        )

        adapter._log_metric(metric)

    def test_duration_patterns(self, adapter):
        mock_histogram = Mock()
        adapter.meter.create_histogram.return_value = mock_histogram

        for name in ["workflow_duration_seconds", "request_duration_seconds", "processing_duration_seconds"]:
            metric = MetricEvent(
                trace_id="123",
                span_id="456",
                metric=name,
                value=1.0,
                timestamp=1234567890.0,
                unit="s",
                attributes={},
                metric_type=MetricType.HISTOGRAM,
            )
            adapter._log_metric(metric)

        assert mock_histogram.record.call_count == 3

    def test_storage_isolation(self, adapter):
        mock_histogram = Mock()
        mock_counter = Mock()
        adapter.meter.create_histogram.return_value = mock_histogram
        adapter.meter.create_counter.return_value = mock_counter

        duration_metric = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="test_duration_seconds",
            value=1.0,
            timestamp=1234567890.0,
            unit="s",
            attributes={},
            metric_type=MetricType.HISTOGRAM,
        )
        counter_metric = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="test_counter",
            value=1,
            timestamp=1234567890.0,
            unit="1",
            attributes={},
        )

        adapter._log_metric(duration_metric)
        adapter._log_metric(counter_metric)

        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        assert "test_duration_seconds" in _GLOBAL_STORAGE["histograms"]
        assert "test_counter" in _GLOBAL_STORAGE["counters"]
