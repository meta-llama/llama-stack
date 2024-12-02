# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import threading
from typing import List

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

from llama_stack.distribution.datatypes import Api
from llama_stack.providers.remote.telemetry.opentelemetry.postgres_processor import (
    PostgresSpanProcessor,
)
from llama_stack.providers.utils.telemetry.jaeger import JaegerTraceStore
from llama_stack.providers.utils.telemetry.postgres import PostgresTraceStore


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
    def __init__(self, config: OpenTelemetryConfig, deps) -> None:
        self.config = config
        self.datasetio = deps[Api.datasetio]

        if config.trace_store == "jaeger":
            self.trace_store = JaegerTraceStore(
                config.jaeger_query_endpoint, config.service_name
            )
        elif config.trace_store == "postgres":
            self.trace_store = PostgresTraceStore(config.postgres_conn_string)
        else:
            raise ValueError(f"Invalid trace store: {config.trace_store}")

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
        trace.get_tracer_provider().add_span_processor(
            PostgresSpanProcessor(self.config.postgres_conn_string)
        )
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

                context = trace.Context(trace_id=trace_id)
                if parent_span:
                    context = trace.set_span_in_context(parent_span, context)

                span = tracer.start_span(
                    name=event.payload.name,
                    context=context,
                    attributes=event.attributes or {},
                )
                _GLOBAL_STORAGE["active_spans"][span_id] = span

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
                    span.end()
                    _GLOBAL_STORAGE["active_spans"].pop(span_id, None)

    async def get_trace(self, trace_id: str) -> TraceTree:
        return await self.trace_store.get_trace(trace_id)

    async def get_agent_trace(
        self,
        session_ids: List[str],
    ) -> List[EvalTrace]:
        traces = []
        for session_id in session_ids:
            traces_for_session = await self.trace_store.get_traces_for_sessions(
                [session_id]
            )
            for session_trace in traces_for_session:
                trace_details = await self._get_simplified_agent_trace(
                    session_trace.trace_id, session_id
                )
                traces.extend(trace_details)

        return traces

    async def export_agent_trace(self, session_ids: List[str], dataset_id: str) -> None:
        traces = await self.get_agent_trace(session_ids)
        traces_dict = [
            {
                "step": trace.step,
                "input": trace.input,
                "output": trace.output,
                "session_id": trace.session_id,
            }
            for trace in traces
        ]
        await self.datasetio.upload_rows(dataset_id, traces_dict)

    async def _get_simplified_agent_trace(
        self, trace_id: str, session_id: str
    ) -> List[EvalTrace]:
        trace_tree = await self.get_trace(trace_id)
        if not trace_tree or not trace_tree.root:
            return []

        def find_execute_turn_children(node: SpanNode) -> List[EvalTrace]:
            results = []
            if node.span.name == "create_and_execute_turn":
                # Sort children by start time
                sorted_children = sorted(node.children, key=lambda x: x.span.start_time)
                for child in sorted_children:
                    results.append(
                        EvalTrace(
                            step=child.span.name,
                            input=str(child.span.attributes.get("input", "")),
                            output=str(child.span.attributes.get("output", "")),
                            session_id=session_id,
                            expected_output="",
                        )
                    )

            # Recursively process children
            for child in node.children:
                results.extend(find_execute_turn_children(child))
            return results

        return find_execute_turn_children(trace_tree.root)
