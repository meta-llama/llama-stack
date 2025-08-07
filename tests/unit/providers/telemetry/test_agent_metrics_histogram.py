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
            metric_type=MetricType.COUNTER,  # Fix: Add proper metric type
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
            metric_type=MetricType.HISTOGRAM,  # Fix: Add proper metric type
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
            metric_type=MetricType.COUNTER,  # Fix: Add proper metric type
        )

        adapter._log_metric(duration_metric)
        adapter._log_metric(counter_metric)

        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        assert "test_duration_seconds" in _GLOBAL_STORAGE["histograms"]
        assert "test_counter" in _GLOBAL_STORAGE["counters"]


class TestMetricTypes:
    def setup_method(self):
        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        _GLOBAL_STORAGE.clear()
        for key in ["histograms", "counters", "up_down_counters", "gauges"]:
            _GLOBAL_STORAGE[key] = {}

    def test_histogram_uses_record(self, adapter):
        mock_histogram = Mock()
        adapter.meter.create_histogram.return_value = mock_histogram

        metric = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="duration",
            value=1.5,
            timestamp=1234567890.0,
            unit="s",
            attributes={},
            metric_type=MetricType.HISTOGRAM,
        )
        adapter._log_metric(metric)

        mock_histogram.record.assert_called_once_with(1.5, attributes={})

    def test_counter_uses_add(self, adapter):
        mock_counter = Mock()
        adapter.meter.create_counter.return_value = mock_counter

        metric = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="requests",
            value=1,
            timestamp=1234567890.0,
            unit="1",
            attributes={},
            metric_type=MetricType.COUNTER,
        )
        adapter._log_metric(metric)

        mock_counter.add.assert_called_once_with(1, attributes={})

    def test_up_down_counter_uses_add(self, adapter):
        mock_counter = Mock()
        adapter.meter.create_up_down_counter.return_value = mock_counter

        metric = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="connections",
            value=-2,
            timestamp=1234567890.0,
            unit="1",
            attributes={},
            metric_type=MetricType.UP_DOWN_COUNTER,
        )
        adapter._log_metric(metric)

        mock_counter.add.assert_called_once_with(-2, attributes={})

    def test_gauge_uses_set(self, adapter):
        mock_gauge = Mock()
        adapter.meter.create_gauge.return_value = mock_gauge

        metric = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="cpu_percent",
            value=85.7,
            timestamp=1234567890.0,
            unit="%",
            attributes={},
            metric_type=MetricType.GAUGE,
        )
        adapter._log_metric(metric)

        mock_gauge.set.assert_called_once_with(85.7, attributes={})

    def test_all_types_work(self, adapter):
        mocks = {"histogram": Mock(), "counter": Mock(), "up_down_counter": Mock(), "gauge": Mock()}
        adapter.meter.create_histogram.return_value = mocks["histogram"]
        adapter.meter.create_counter.return_value = mocks["counter"]
        adapter.meter.create_up_down_counter.return_value = mocks["up_down_counter"]
        adapter.meter.create_gauge.return_value = mocks["gauge"]

        test_data = [
            (MetricType.HISTOGRAM, 1.5, mocks["histogram"], "record"),
            (MetricType.COUNTER, 5, mocks["counter"], "add"),
            (MetricType.UP_DOWN_COUNTER, -3, mocks["up_down_counter"], "add"),
            (MetricType.GAUGE, 42.0, mocks["gauge"], "set"),
        ]

        for metric_type, value, mock_obj, method_name in test_data:
            metric = MetricEvent(
                trace_id="123",
                span_id="456",
                metric="test",
                value=value,
                timestamp=1234567890.0,
                unit="1",
                attributes={},
                metric_type=metric_type,
            )
            adapter._log_metric(metric)
            getattr(mock_obj, method_name).assert_called_with(value, attributes={})
