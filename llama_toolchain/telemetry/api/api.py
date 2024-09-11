# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Literal, Optional, Protocol, Union

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel, Field
from typing_extensions import Annotated


@json_schema_type
class SpanStatus(Enum):
    OK = "ok"
    ERROR = "error"


@json_schema_type
class Span(BaseModel):
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    attributes: Optional[Dict[str, Any]] = Field(default_factory=dict)


@json_schema_type
class Trace(BaseModel):
    trace_id: str
    root_span_id: str
    start_time: datetime
    end_time: Optional[datetime] = None


@json_schema_type
class EventType(Enum):
    UNSTRUCTURED_LOG = "unstructured_log"

    # all structured log events below
    SPAN_START = "span_start"
    SPAN_END = "span_end"
    METRIC = "metric"

    def is_structured(self) -> bool:
        return self != EventType.UNSTRUCTURED_LOG


@json_schema_type
class LogSeverity(Enum):
    VERBOSE = "verbose"
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    CRITICAL = "critical"


class EventCommon(BaseModel):
    trace_id: str
    span_id: str
    timestamp: datetime
    attributes: Optional[Dict[str, Any]] = Field(default_factory=dict)


@json_schema_type
class LoggingEvent(EventCommon):
    type: Literal[EventType.UNSTRUCTURED_LOG.value] = EventType.UNSTRUCTURED_LOG.value
    message: str
    severity: LogSeverity


@json_schema_type
class SpanStartEvent(EventCommon):
    type: Literal[EventType.SPAN_START.value] = EventType.SPAN_START.value
    name: str
    parent_span_id: Optional[str] = None


@json_schema_type
class SpanEndEvent(EventCommon):
    type: Literal[EventType.SPAN_END.value] = EventType.SPAN_END.value
    status: SpanStatus


@json_schema_type
class MetricEvent(EventCommon):
    type: Literal[EventType.METRIC.value] = EventType.METRIC.value
    metric: str  # this would be an enum
    value: Union[int, float]
    unit: str


Event = Annotated[
    Union[LoggingEvent, SpanStartEvent, SpanEndEvent],
    Field(discriminator="type"),
]


class Telemetry(Protocol):
    @webmethod(route="/telemetry/log_event")
    async def log_event(self, event: Event): ...

    @webmethod(route="/telemetry/get_trace", method="GET")
    async def get_trace(self, trace_id: str) -> Trace: ...
