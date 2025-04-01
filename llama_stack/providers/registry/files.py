# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.providers.datatypes import (
    AdapterSpec,
    Api,
    ProviderSpec,
    remote_provider_spec,
)


def available_providers() -> List[ProviderSpec]:
    return [
        remote_provider_spec(
            api=Api.files,
            adapter=AdapterSpec(
                adapter_type="s3",
                pip_packages=["aioboto3"],
                module="llama_stack.providers.remote.files.object.s3",
                config_class="llama_stack.providers.remote.files.object.s3.config.S3ImplConfig",
                provider_data_validator="llama_stack.providers.remote.files.object.s3.S3ProviderDataValidator",
            ),
        ),
    ]
