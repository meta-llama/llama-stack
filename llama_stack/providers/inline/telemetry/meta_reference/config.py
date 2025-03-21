# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator

from llama_stack.distribution.utils.config_dirs import RUNTIME_BASE_DIR


class TelemetrySink(str, Enum):
    OTEL_TRACE = "otel_trace"
    OTEL_METRIC = "otel_metric"
    SQLITE = "sqlite"
    CONSOLE = "console"


class TelemetryConfig(BaseModel):
    otel_trace_endpoint: str = Field(
        default="http://localhost:4318/v1/traces",
        description="The OpenTelemetry collector endpoint URL for traces",
    )
    otel_metric_endpoint: str = Field(
        default="http://localhost:4318/v1/metrics",
        description="The OpenTelemetry collector endpoint URL for metrics",
    )
    sinks: List[TelemetrySink] = Field(
        default=[TelemetrySink.CONSOLE, TelemetrySink.SQLITE],
        description="List of telemetry sinks to enable (possible values: otel, sqlite, console)",
    )
    sqlite_db_path: str = Field(
        default=(RUNTIME_BASE_DIR / "trace_store.db").as_posix(),
        description="The path to the SQLite database to use for storing traces",
    )

    @field_validator("sinks", mode="before")
    @classmethod
    def validate_sinks(cls, v):
        if isinstance(v, str):
            return [TelemetrySink(sink.strip()) for sink in v.split(",")]
        return v

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, db_name: str = "trace_store.db") -> Dict[str, Any]:
        return {
            "sinks": "${env.TELEMETRY_SINKS:console,sqlite}",
            "sqlite_db_path": "${env.SQLITE_DB_PATH:" + __distro_dir__ + "/" + db_name + "}",
        }
