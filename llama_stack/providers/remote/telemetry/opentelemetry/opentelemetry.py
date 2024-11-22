# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import threading
from datetime import datetime

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

from llama_stack.providers.utils.telemetry.tracing import generate_short_uuid

from llama_stack.apis.telemetry import *  # noqa: F403

from .config import OpenTelemetryConfig


def string_to_trace_id(s: str) -> int:
    # Convert the string to bytes and then to an integer
    return int.from_bytes(s.encode(), byteorder="big", signed=False)


def string_to_span_id(s: str) -> int:
    # Use only the first 8 bytes (64 bits) for span ID
    return int.from_bytes(s.encode()[:8], byteorder="big", signed=False)


def is_tracing_enabled(tracer):
    with tracer.start_as_current_span("check_tracing") as span:
        return span.is_recording()


class OpenTelemetryAdapter(Telemetry):
    def __init__(self, config: OpenTelemetryConfig):
        self.config = config

        resource = Resource.create(
            {
                ResourceAttributes.SERVICE_NAME: self.config.service_name,
            }
        )

        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        otlp_exporter = OTLPSpanExporter(
            endpoint=self.config.otel_endpoint,
        )
        span_processor = BatchSpanProcessor(otlp_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        # Set up metrics
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint=self.config.otel_endpoint,
            )
        )
        metric_provider = MeterProvider(
            resource=resource, metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(metric_provider)
        self.meter = metrics.get_meter(__name__)
        # Initialize metric storage
        self._counters = {}
        self._gauges = {}
        self._up_down_counters = {}
        self._active_spans = {}
        self._lock = threading.Lock()

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        trace.get_tracer_provider().force_flush()
        trace.get_tracer_provider().shutdown()
        metrics.get_meter_provider().shutdown()

    async def log_event(self, event: Event) -> None:
        if isinstance(event, UnstructuredLogEvent):
            self._log_unstructured(event)
        elif isinstance(event, MetricEvent):
            self._log_metric(event)
        elif isinstance(event, StructuredLogEvent):
            self._log_structured(event)

    def _log_unstructured(self, event: UnstructuredLogEvent) -> None:
        with self._lock:
            # Check if there's an existing span in the cache
            span_id = string_to_span_id(event.span_id)
            span = self._active_spans.get(span_id)

            if span:
                # Use existing span
                timestamp_ns = int(event.timestamp.timestamp() * 1e9)
                span.add_event(
                    name=event.message,
                    attributes={"severity": event.severity.value, **event.attributes},
                    timestamp=timestamp_ns,
                )
            else:
                print(
                    f"Warning: No active span found for span_id {span_id}. Dropping event: {event}"
                )

    def _get_or_create_counter(self, name: str, unit: str) -> metrics.Counter:
        if name not in self._counters:
            self._counters[name] = self.meter.create_counter(
                name=name,
                unit=unit,
                description=f"Counter for {name}",
            )
        return self._counters[name]

    def _get_or_create_gauge(self, name: str, unit: str) -> metrics.ObservableGauge:
        if name not in self._gauges:
            self._gauges[name] = self.meter.create_gauge(
                name=name,
                unit=unit,
                description=f"Gauge for {name}",
            )
        return self._gauges[name]

    def _log_metric(self, event: MetricEvent) -> None:
        if isinstance(event.value, int):
            counter = self._get_or_create_counter(event.metric, event.unit)
            counter.add(event.value, attributes=event.attributes)
        elif isinstance(event.value, float):
            up_down_counter = self._get_or_create_up_down_counter(
                event.metric, event.unit
            )
            up_down_counter.add(event.value, attributes=event.attributes)

    def _get_or_create_up_down_counter(
        self, name: str, unit: str
    ) -> metrics.UpDownCounter:
        if name not in self._up_down_counters:
            self._up_down_counters[name] = self.meter.create_up_down_counter(
                name=name,
                unit=unit,
                description=f"UpDownCounter for {name}",
            )
        return self._up_down_counters[name]

    def _log_structured(self, event: StructuredLogEvent) -> None:
        with self._lock:
            trace_id = string_to_trace_id(event.trace_id)
            span_id = string_to_span_id(event.span_id)

            tracer = trace.get_tracer(__name__)
            span_context = trace.SpanContext(
                trace_id=trace_id,
                span_id=span_id,
                is_remote=True,
                trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
                trace_state=trace.TraceState(),
            )

            if isinstance(event.payload, SpanStartPayload):
                # Get parent span if it exists
                parent_span = None
                for active_span in self._active_spans.values():
                    if active_span.is_recording():
                        parent_span = active_span
                        break

                # Create the context properly
                context = trace.Context()
                if parent_span:
                    context = trace.set_span_in_context(parent_span)

                span = tracer.start_span(
                    name=event.payload.name,
                    context=context,
                    attributes=event.attributes or {},
                    start_time=int(event.timestamp.timestamp() * 1e9),
                )
                self._active_spans[span_id] = span

                # Set the span as current
                _ = trace.set_span_in_context(span)
                trace.use_span(span, end_on_exit=False)

            elif isinstance(event.payload, SpanEndPayload):
                # Retrieve and end the existing span
                span = self._active_spans.get(span_id)
                if span:
                    if event.attributes:
                        span.set_attributes(event.attributes)

                    status = (
                        trace.Status(status_code=trace.StatusCode.OK)
                        if event.payload.status == SpanStatus.OK
                        else trace.Status(status_code=trace.StatusCode.ERROR)
                    )
                    span.set_status(status)
                    span.end(end_time=int(event.timestamp.timestamp() * 1e9))

                    # Remove from active spans
                    del self._active_spans[span_id]

    async def get_trace(self, trace_id: str) -> Trace:
        raise NotImplementedError("Trace retrieval not implemented yet")


# Usage example
async def main():
    telemetry = OpenTelemetryAdapter(OpenTelemetryConfig())
    await telemetry.initialize()

    # # Log a metric event
    # await telemetry.log_event(
    #     MetricEvent(
    #         trace_id="trace123",
    #         span_id="span456",
    #         timestamp=datetime.now(),
    #         metric="my_metric",
    #         value=42,
    #         unit="count",
    #     )
    # )

    # Log a structured event (span start)
    trace_id = generate_short_uuid(16)
    span_id = generate_short_uuid(8)
    span_id_2 = generate_short_uuid(8)
    await telemetry.log_event(
        StructuredLogEvent(
            trace_id=trace_id,
            span_id=span_id,
            timestamp=datetime.now(),
            payload=SpanStartPayload(name="my_operation"),
        )
    )

    # Log an unstructured event
    await telemetry.log_event(
        UnstructuredLogEvent(
            trace_id=trace_id,
            span_id=span_id,
            timestamp=datetime.now(),
            message="This is a log message",
            severity=LogSeverity.INFO,
        )
    )

    await telemetry.log_event(
        StructuredLogEvent(
            trace_id=trace_id,
            span_id=span_id_2,
            timestamp=datetime.now(),
            payload=SpanStartPayload(name="my_operation_2"),
        )
    )
    await telemetry.log_event(
        UnstructuredLogEvent(
            trace_id=trace_id,
            span_id=span_id_2,
            timestamp=datetime.now(),
            message="This is a log message 2",
            severity=LogSeverity.INFO,
        )
    )

    await telemetry.log_event(
        StructuredLogEvent(
            trace_id=trace_id,
            span_id=span_id_2,
            timestamp=datetime.now(),
            payload=SpanEndPayload(status=SpanStatus.OK),
        )
    )
    await telemetry.log_event(
        StructuredLogEvent(
            trace_id=trace_id,
            span_id=span_id,
            timestamp=datetime.now(),
            payload=SpanEndPayload(status=SpanStatus.OK),
        )
    )

    await telemetry.shutdown()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
