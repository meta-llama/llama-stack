# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from pydantic import BaseModel, Field


class OpenTelemetryConfig(BaseModel):
    otel_endpoint: str = Field(
        default="http://localhost:4318/v1/traces",
        description="The OpenTelemetry collector endpoint URL",
    )
    service_name: str = Field(
        default="llama-stack",
        description="The service name to use for telemetry",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> Dict[str, Any]:
        return {
            "otel_endpoint": "${env.OTEL_ENDPOINT:http://localhost:4318/v1/traces}",
            "service_name": "${env.OTEL_SERVICE_NAME:llama-stack}",
        }
