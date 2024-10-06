# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime

from opentelemetry import metrics, trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

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

        self.resource = Resource.create(
            {ResourceAttributes.SERVICE_NAME: "foobar-service"}
        )

        # Set up tracing with Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=self.config.jaeger_host,
            agent_port=self.config.jaeger_port,
        )
        trace_provider = TracerProvider(resource=self.resource)
        trace_processor = BatchSpanProcessor(jaeger_exporter)
        trace_provider.add_span_processor(trace_processor)
        trace.set_tracer_provider(trace_provider)
        self.tracer = trace.get_tracer(__name__)

        # Set up metrics
        metric_reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
        metric_provider = MeterProvider(
            resource=self.resource, metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(metric_provider)
        self.meter = metrics.get_meter(__name__)

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
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
        span = trace.get_current_span()
        span.add_event(
            name=event.message,
            attributes={"severity": event.severity.value, **event.attributes},
            timestamp=event.timestamp,
        )

    def _log_metric(self, event: MetricEvent) -> None:
        if isinstance(event.value, int):
            self.meter.create_counter(
                name=event.metric,
                unit=event.unit,
                description=f"Counter for {event.metric}",
            ).add(event.value, attributes=event.attributes)
        elif isinstance(event.value, float):
            self.meter.create_gauge(
                name=event.metric,
                unit=event.unit,
                description=f"Gauge for {event.metric}",
            ).set(event.value, attributes=event.attributes)

    def _log_structured(self, event: StructuredLogEvent) -> None:
        if isinstance(event.payload, SpanStartPayload):
            context = trace.set_span_in_context(
                trace.NonRecordingSpan(
                    trace.SpanContext(
                        trace_id=string_to_trace_id(event.trace_id),
                        span_id=string_to_span_id(event.span_id),
                        is_remote=True,
                    )
                )
            )
            span = self.tracer.start_span(
                name=event.payload.name,
                kind=trace.SpanKind.INTERNAL,
                context=context,
                attributes=event.attributes,
            )

            if event.payload.parent_span_id:
                span.set_parent(
                    trace.SpanContext(
                        trace_id=string_to_trace_id(event.trace_id),
                        span_id=string_to_span_id(event.payload.parent_span_id),
                        is_remote=True,
                    )
                )
        elif isinstance(event.payload, SpanEndPayload):
            span = trace.get_current_span()
            span.set_status(
                trace.Status(
                    trace.StatusCode.OK
                    if event.payload.status == SpanStatus.OK
                    else trace.StatusCode.ERROR
                )
            )
            span.end(end_time=event.timestamp)

    async def get_trace(self, trace_id: str) -> Trace:
        # we need to look up the root span id
        raise NotImplementedError("not yet no")


# Usage example
async def main():
    telemetry = OpenTelemetryTelemetry("my-service")
    await telemetry.initialize()

    # Log an unstructured event
    await telemetry.log_event(
        UnstructuredLogEvent(
            trace_id="trace123",
            span_id="span456",
            timestamp=datetime.now(),
            message="This is a log message",
            severity=LogSeverity.INFO,
        )
    )

    # Log a metric event
    await telemetry.log_event(
        MetricEvent(
            trace_id="trace123",
            span_id="span456",
            timestamp=datetime.now(),
            metric="my_metric",
            value=42,
            unit="count",
        )
    )

    # Log a structured event (span start)
    await telemetry.log_event(
        StructuredLogEvent(
            trace_id="trace123",
            span_id="span789",
            timestamp=datetime.now(),
            payload=SpanStartPayload(name="my_operation"),
        )
    )

    # Log a structured event (span end)
    await telemetry.log_event(
        StructuredLogEvent(
            trace_id="trace123",
            span_id="span789",
            timestamp=datetime.now(),
            payload=SpanEndPayload(status=SpanStatus.OK),
        )
    )

    await telemetry.shutdown()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
