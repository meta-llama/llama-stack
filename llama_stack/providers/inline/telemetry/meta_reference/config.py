# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from llama_stack.distribution.utils.config_dirs import RUNTIME_BASE_DIR


class TelemetrySink(str, Enum):
    OTEL = "otel"
    SQLITE = "sqlite"
    CONSOLE = "console"


class MetricsStoreType(str, Enum):
    PROMETHEUS = "prometheus"


class PrometheusConfig(BaseModel):
    endpoint: str = Field(
        default="http://localhost:9090",
        description="The Prometheus endpoint URL",
    )
    disable_ssl: bool = Field(
        default=True,
        description="Whether to disable SSL for the Prometheus endpoint",
    )


class PrometheusMetricsStoreConfig(BaseModel):
    type: Literal["prometheus"] = Field(
        default="prometheus",
        description="The type of metrics store to use",
    )
    endpoint: str = Field(
        default="http://localhost:9090",
        description="The Prometheus endpoint URL",
    )
    disable_ssl: bool = Field(
        default=True,
        description="Whether to disable SSL for the Prometheus endpoint",
    )


MetricsStoreConfig = Annotated[Union[PrometheusMetricsStoreConfig], Field(discriminator="type")]


class TelemetryConfig(BaseModel):
    otel_endpoint: str = Field(
        default="http://localhost:4318",
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
    metrics_store_config: Optional[MetricsStoreConfig] = None

    @field_validator("sinks", mode="before")
    @classmethod
    def validate_sinks(cls, v):
        if isinstance(v, str):
            return [TelemetrySink(sink.strip()) for sink in v.split(",")]
        return v

    @classmethod
    def sample_run_config(cls, __distro_dir__: str = "runtime", db_name: str = "trace_store.db") -> Dict[str, Any]:
        return {
            "service_name": "${env.OTEL_SERVICE_NAME:llama-stack}",
            "sinks": "${env.TELEMETRY_SINKS:console,sqlite}",
            "sqlite_db_path": "${env.SQLITE_DB_PATH:~/.llama/" + __distro_dir__ + "/" + db_name + "}",
        }
