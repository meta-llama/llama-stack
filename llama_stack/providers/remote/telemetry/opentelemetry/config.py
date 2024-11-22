# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel


class OpenTelemetryConfig(BaseModel):
    otel_endpoint: str = "http://localhost:4318/v1/traces"
    service_name: str = "llama-stack"
