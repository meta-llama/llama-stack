# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import threading
from typing import Any, Dict, List, Optional

import aiohttp

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

    async def get_traces_for_eval(
        self,
        session_ids: List[str],
        lookback: str = "1h",
        limit: int = 100,
        dataset_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        traces = []

        # Fetch traces for each session ID individually
        for session_id in session_ids:
            params = {
                "service": self.config.service_name,
                "lookback": lookback,
                "limit": limit,
                "tags": f'{{"session_id":"{session_id}"}}',
            }

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self.config.export_endpoint, params=params
                    ) as response:
                        if response.status != 200:
                            raise Exception(
                                f"Failed to query Jaeger: {response.status} {await response.text()}"
                            )

                        traces_data = await response.json()
                        seen_trace_ids = set()

                        # For each trace ID, get the detailed trace information
                        for trace_data in traces_data.get("data", []):
                            trace_id = trace_data.get("traceID")
                            if trace_id and trace_id not in seen_trace_ids:
                                seen_trace_ids.add(trace_id)
                                trace_details = await self.get_trace_for_eval(trace_id)
                                if trace_details:
                                    traces.append(trace_details)

            except Exception as e:
                raise Exception(f"Error querying Jaeger traces: {str(e)}") from e

        return traces

    async def get_trace(self, trace_id: str) -> Dict[str, Any]:
        params = {
            "traceID": trace_id,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.export_endpoint}/{trace_id}", params=params
                ) as response:
                    if response.status != 200:
                        raise Exception(
                            f"Failed to query Jaeger: {response.status} {await response.text()}"
                        )

                    trace_data = await response.json()
                    if not trace_data.get("data") or not trace_data["data"]:
                        return None

                    # First pass: Build span map
                    span_map = {}
                    for span in trace_data["data"][0]["spans"]:
                        start_time = span["startTime"]
                        end_time = start_time + span.get(
                            "duration", 0
                        )  # Get end time from duration if available

                        # Some systems store end time directly in the span
                        if "endTime" in span:
                            end_time = span["endTime"]
                            duration = end_time - start_time
                        else:
                            duration = span.get("duration", 0)

                        span_map[span["spanID"]] = {
                            "id": span["spanID"],
                            "name": span["operationName"],
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration": duration,
                            "tags": {
                                tag["key"]: tag["value"] for tag in span.get("tags", [])
                            },
                            "children": [],
                        }

                    # Second pass: Build parent-child relationships
                    root_spans = []
                    for span in trace_data["data"][0]["spans"]:
                        references = span.get("references", [])
                        if references and references[0]["refType"] == "CHILD_OF":
                            parent_id = references[0]["spanID"]
                            if parent_id in span_map:
                                span_map[parent_id]["children"].append(
                                    span_map[span["spanID"]]
                                )
                        else:
                            root_spans.append(span_map[span["spanID"]])

                    return {
                        "trace_id": trace_id,
                        "spans": root_spans,
                    }

        except Exception as e:
            raise Exception(f"Error querying Jaeger trace structure: {str(e)}") from e

    async def get_trace_for_eval(self, trace_id: str) -> List[Dict[str, Any]]:
        """
        Get simplified trace information focusing on first-level children of create_and_execute_turn operations.
        Returns a list of spans with name, input, and output information, sorted by start time.
        """
        trace_data = await self.get_trace(trace_id)
        if not trace_data:
            return []

        def find_execute_turn_children(
            spans: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            results = []
            for span in spans:
                if span["name"] == "create_and_execute_turn":
                    # Extract and format children spans
                    children = sorted(span["children"], key=lambda x: x["start_time"])
                    for child in children:
                        results.append(
                            {
                                "name": child["name"],
                                "input": child["tags"].get("input", ""),
                                "output": child["tags"].get("output", ""),
                            }
                        )
                # Recursively search in children
                results.extend(find_execute_turn_children(span["children"]))
            return results

        return find_execute_turn_children(trace_data["spans"])
