# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import threading

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes


from llama_stack.apis.telemetry import *  # noqa: F403

from .config import OpenTelemetryConfig

_GLOBAL_STORAGE = {
    "active_spans": {},
    "counters": {},
    "gauges": {},
    "up_down_counters": {},
}
_global_lock = threading.Lock()


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
        self._lock = _global_lock

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
            # Use global storage instead of instance storage
            span_id = string_to_span_id(event.span_id)
            span = _GLOBAL_STORAGE["active_spans"].get(span_id)

            if span:
                timestamp_ns = int(event.timestamp.timestamp() * 1e9)
                span.add_event(
                    name=event.type,
                    attributes={
                        "message": event.message,
                        "severity": event.severity.value,
                        **event.attributes,
                    },
                    timestamp=timestamp_ns,
                )
            else:
                print(
                    f"Warning: No active span found for span_id {span_id}. Dropping event: {event}"
                )

    def _get_or_create_counter(self, name: str, unit: str) -> metrics.Counter:
        if name not in _GLOBAL_STORAGE["counters"]:
            _GLOBAL_STORAGE["counters"][name] = self.meter.create_counter(
                name=name,
                unit=unit,
                description=f"Counter for {name}",
            )
        return _GLOBAL_STORAGE["counters"][name]

    def _get_or_create_gauge(self, name: str, unit: str) -> metrics.ObservableGauge:
        if name not in _GLOBAL_STORAGE["gauges"]:
            _GLOBAL_STORAGE["gauges"][name] = self.meter.create_gauge(
                name=name,
                unit=unit,
                description=f"Gauge for {name}",
            )
        return _GLOBAL_STORAGE["gauges"][name]

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
        if name not in _GLOBAL_STORAGE["up_down_counters"]:
            _GLOBAL_STORAGE["up_down_counters"][name] = (
                self.meter.create_up_down_counter(
                    name=name,
                    unit=unit,
                    description=f"UpDownCounter for {name}",
                )
            )
        return _GLOBAL_STORAGE["up_down_counters"][name]

    def _log_structured(self, event: StructuredLogEvent) -> None:
        with self._lock:
            span_id = string_to_span_id(event.span_id)
            trace_id = string_to_trace_id(event.trace_id)
            tracer = trace.get_tracer(__name__)

            if isinstance(event.payload, SpanStartPayload):
                # Check if span already exists to prevent duplicates
                if span_id in _GLOBAL_STORAGE["active_spans"]:
                    return

                parent_span = None
                if event.payload.parent_span_id:
                    parent_span_id = string_to_span_id(event.payload.parent_span_id)
                    parent_span = _GLOBAL_STORAGE["active_spans"].get(parent_span_id)

                # Create a new trace context with the trace_id
                context = trace.Context(trace_id=trace_id)
                if parent_span:
                    context = trace.set_span_in_context(parent_span, context)

                span = tracer.start_span(
                    name=event.payload.name,
                    context=context,
                    attributes=event.attributes or {},
                    start_time=int(event.timestamp.timestamp() * 1e9),
                )
                _GLOBAL_STORAGE["active_spans"][span_id] = span

                # Set as current span using context manager
                with trace.use_span(span, end_on_exit=False):
                    pass  # Let the span continue beyond this block

            elif isinstance(event.payload, SpanEndPayload):
                span = _GLOBAL_STORAGE["active_spans"].get(span_id)
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
                    _GLOBAL_STORAGE["active_spans"].pop(span_id, None)

    async def get_trace(self, trace_id: str) -> Trace:
        raise NotImplementedError("Trace retrieval not implemented yet")
