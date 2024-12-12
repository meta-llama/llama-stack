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
            provider_type="inline::meta-reference",
            pip_packages=[
                "opentelemetry-sdk",
                "opentelemetry-exporter-otlp-proto-http",
            ],
            api_dependencies=[Api.datasetio],
            module="llama_stack.providers.inline.telemetry.meta_reference",
            config_class="llama_stack.providers.inline.telemetry.meta_reference.config.TelemetryConfig",
        ),
        remote_provider_spec(
            api=Api.telemetry,
            adapter=AdapterSpec(
                adapter_type="sample",
                pip_packages=[],
                module="llama_stack.providers.remote.telemetry.sample",
                config_class="llama_stack.providers.remote.telemetry.sample.SampleConfig",
            ),
        ),
    ]
