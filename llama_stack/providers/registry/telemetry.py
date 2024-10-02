# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.distribution.datatypes import *  # noqa: F403


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.telemetry,
            provider_type="meta-reference",
            pip_packages=[],
            module="llama_stack.providers.impls.meta_reference.telemetry",
            config_class="llama_stack.providers.impls.meta_reference.telemetry.ConsoleConfig",
        ),
        remote_provider_spec(
            api=Api.telemetry,
            adapter=AdapterSpec(
                adapter_type="sample",
                pip_packages=[],
                module="llama_stack.providers.adapters.telemetry.sample",
                config_class="llama_stack.providers.adapters.telemetry.sample.SampleConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.telemetry,
            adapter=AdapterSpec(
                adapter_type="opentelemetry-jaeger",
                pip_packages=[
                    "opentelemetry-api",
                    "opentelemetry-sdk",
                    "opentelemetry-exporter-jaeger",
                    "opentelemetry-semantic-conventions",
                ],
                module="llama_stack.providers.adapters.telemetry.opentelemetry",
                config_class="llama_stack.providers.adapters.telemetry.opentelemetry.OpenTelemetryConfig",
            ),
        ),
    ]
