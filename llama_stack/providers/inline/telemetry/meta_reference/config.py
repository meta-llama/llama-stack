# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from llama_stack.distribution.utils.config_dirs import RUNTIME_BASE_DIR


class TelemetrySink(str, Enum):
    OTEL = "otel"
    SQLITE = "sqlite"
    CONSOLE = "console"


class TelemetryConfig(BaseModel):
    otel_endpoint: str = Field(
        default="http://localhost:4318/v1/traces",
        description="The OpenTelemetry collector endpoint URL",
    )
    service_name: str = Field(
        default="llama-stack",
        description="The service name to use for telemetry",
    )
    sinks: List[TelemetrySink] = Field(
        default=[TelemetrySink.CONSOLE, TelemetrySink.SQLITE],
        description="List of telemetry sinks to enable (possible values: otel, sqlite, console)",
    )
    sqlite_db_path: str = Field(
        default=(RUNTIME_BASE_DIR / "trace_store.db").as_posix(),
        description="The path to the SQLite database to use for storing traces",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> Dict[str, Any]:
        return {
            "service_name": "${env.OTEL_SERVICE_NAME:llama-stack}",
            "sinks": "${env.TELEMETRY_SINKS:['console', 'sqlite']}",
            "sqlite_db_path": "${env.SQLITE_DB_PATH:${runtime.base_dir}/trace_store.db}",
        }
