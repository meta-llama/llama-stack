# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import threading
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes
from prometheus_api_client import PrometheusConnect

from llama_stack.apis.telemetry import (
    Event,
    GetMetricsResponse,
    MetricDataPoint,
    MetricEvent,
    MetricLabelMatcher,
    MetricQueryType,
    MetricSeries,
    QueryCondition,
    QuerySpanTreeResponse,
    QueryTracesResponse,
    Span,
    SpanEndPayload,
    SpanStartPayload,
    SpanStatus,
    StructuredLogEvent,
    Telemetry,
    Trace,
    UnstructuredLogEvent,
)
from llama_stack.distribution.datatypes import Api
from llama_stack.providers.inline.telemetry.meta_reference.console_span_processor import (
    ConsoleSpanProcessor,
)
from llama_stack.providers.inline.telemetry.meta_reference.sqlite_span_processor import (
    SQLiteSpanProcessor,
)
from llama_stack.providers.utils.telemetry.dataset_mixin import TelemetryDatasetMixin
from llama_stack.providers.utils.telemetry.sqlite_trace_store import SQLiteTraceStore

from .config import TelemetryConfig, TelemetrySink

_GLOBAL_STORAGE = {
    "active_spans": {},
    "counters": {},
    "gauges": {},
    "up_down_counters": {},
}
_global_lock = threading.Lock()
_TRACER_PROVIDER = None


def string_to_trace_id(s: str) -> int:
    # Convert the string to bytes and then to an integer
    return int.from_bytes(s.encode(), byteorder="big", signed=False)


def string_to_span_id(s: str) -> int:
    # Use only the first 8 bytes (64 bits) for span ID
    return int.from_bytes(s.encode()[:8], byteorder="big", signed=False)


def is_tracing_enabled(tracer):
    with tracer.start_as_current_span("check_tracing") as span:
        return span.is_recording()


class TelemetryAdapter(TelemetryDatasetMixin, Telemetry):
    def __init__(self, config: TelemetryConfig, deps: Dict[str, Any]) -> None:
        self.config = config
        self.datasetio_api = deps.get(Api.datasetio)

        resource = Resource.create(
            {
                ResourceAttributes.SERVICE_NAME: self.config.service_name,
            }
        )

        global _TRACER_PROVIDER
        # Initialize the correct span processor based on the provider state.
        # This is needed since once the span processor is set, it cannot be unset.
        # Recreating the telemetry adapter multiple times will result in duplicate span processors.
        # Since the library client can be recreated multiple times in a notebook,
        # the kernel will hold on to the span processor and cause duplicate spans to be written.
        if _TRACER_PROVIDER is None:
            provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(provider)
            _TRACER_PROVIDER = provider
            if TelemetrySink.OTEL in self.config.sinks:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=urljoin(self.config.otel_endpoint, "v1/traces"),
                )
                span_processor = BatchSpanProcessor(otlp_exporter)
                trace.get_tracer_provider().add_span_processor(span_processor)
                metric_reader = PeriodicExportingMetricReader(
                    OTLPMetricExporter(
                        endpoint=urljoin(self.config.otel_endpoint, "v1/metrics"),
                    )
                )
                metric_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
                metrics.set_meter_provider(metric_provider)
            if TelemetrySink.SQLITE in self.config.sinks:
                trace.get_tracer_provider().add_span_processor(SQLiteSpanProcessor(self.config.sqlite_db_path))
            if TelemetrySink.CONSOLE in self.config.sinks:
                trace.get_tracer_provider().add_span_processor(ConsoleSpanProcessor())

        if TelemetrySink.OTEL in self.config.sinks:
            self.meter = metrics.get_meter(__name__)
            self.prom = PrometheusConnect(
                url=self.config.prometheus_endpoint, disable_ssl=self.config.prometheus_disable_ssl
            )
        if TelemetrySink.SQLITE in self.config.sinks:
            self.trace_store = SQLiteTraceStore(self.config.sqlite_db_path)

        self._lock = _global_lock

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        trace.get_tracer_provider().force_flush()

    async def log_event(self, event: Event, ttl_seconds: int = 604800) -> None:
        if isinstance(event, UnstructuredLogEvent):
            self._log_unstructured(event, ttl_seconds)
        elif isinstance(event, MetricEvent):
            self._log_metric(event)
        elif isinstance(event, StructuredLogEvent):
            self._log_structured(event, ttl_seconds)
        else:
            raise ValueError(f"Unknown event type: {event}")

    def _log_unstructured(self, event: UnstructuredLogEvent, ttl_seconds: int) -> None:
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
                        "__ttl__": ttl_seconds,
                        **event.attributes,
                    },
                    timestamp=timestamp_ns,
                )
            else:
                print(f"Warning: No active span found for span_id {span_id}. Dropping event: {event}")

    def _get_or_create_counter(self, name: str, unit: str) -> metrics.Counter:
        if name not in _GLOBAL_STORAGE["counters"]:
            _GLOBAL_STORAGE["counters"][name] = self.meter.create_counter(
                name=name,
                unit=unit,
                description=f"Counter for {name}",
            )
        return _GLOBAL_STORAGE["counters"][name]

    def _log_metric(self, event: MetricEvent) -> None:
        counter = self._get_or_create_counter(event.metric, event.unit)
        counter.add(event.value, attributes=event.attributes)

    def _log_structured(self, event: StructuredLogEvent, ttl_seconds: int) -> None:
        with self._lock:
            span_id = string_to_span_id(event.span_id)
            trace_id = string_to_trace_id(event.trace_id)
            tracer = trace.get_tracer(__name__)
            if event.attributes is None:
                event.attributes = {}
            event.attributes["__ttl__"] = ttl_seconds

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
            else:
                raise ValueError(f"Unknown structured log event: {event}")

    async def query_traces(
        self,
        attribute_filters: Optional[List[QueryCondition]] = None,
        limit: Optional[int] = 100,
        offset: Optional[int] = 0,
        order_by: Optional[List[str]] = None,
    ) -> QueryTracesResponse:
        return QueryTracesResponse(
            data=await self.trace_store.query_traces(
                attribute_filters=attribute_filters,
                limit=limit,
                offset=offset,
                order_by=order_by,
            )
        )

    async def get_trace(self, trace_id: str) -> Trace:
        return await self.trace_store.get_trace(trace_id)

    async def get_span(self, trace_id: str, span_id: str) -> Span:
        return await self.trace_store.get_span(trace_id, span_id)

    async def get_span_tree(
        self,
        span_id: str,
        attributes_to_return: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
    ) -> QuerySpanTreeResponse:
        return QuerySpanTreeResponse(
            data=await self.trace_store.get_span_tree(
                span_id=span_id,
                attributes_to_return=attributes_to_return,
                max_depth=max_depth,
            )
        )

    async def get_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        step: Optional[str] = "15s",
        query_type: MetricQueryType = MetricQueryType.RANGE,
        label_matchers: Optional[List[MetricLabelMatcher]] = None,
    ) -> GetMetricsResponse:
        if TelemetrySink.OTEL not in self.config.sinks:
            return GetMetricsResponse(data=[])

        try:
            # Build query with label matchers if provided
            query = metric_name
            if label_matchers:
                matchers = [f'{m.name}{m.operator.value}"{m.value}"' for m in label_matchers]
                query = f"{metric_name}{{{','.join(matchers)}}}"

            # Use instant query for current values, range query for historical data
            if query_type == MetricQueryType.INSTANT:
                result = self.prom.custom_query(query=query)
                # Convert instant query results to same format as range query
                result = [{"metric": r["metric"], "values": [[r["value"][0], r["value"][1]]]} for r in result]
            else:
                result = self.prom.custom_query_range(
                    query=query,
                    start_time=start_time,
                    end_time=end_time if end_time else None,
                    step=step,
                )

            series = []
            for metric_data in result:
                values = [
                    MetricDataPoint(timestamp=datetime.fromtimestamp(point[0]), value=float(point[1]))
                    for point in metric_data["values"]
                ]
                series.append(MetricSeries(metric=metric_name, labels=metric_data.get("metric", {}), values=values))

            return GetMetricsResponse(data=series)

        except Exception as e:
            print(f"Error querying metrics: {e}")
            return GetMetricsResponse(data=[])
