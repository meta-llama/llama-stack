# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.datatypes import (
    AdapterSpec,
    Api,
    ProviderSpec,
    remote_provider_spec,
)
from llama_stack.providers.utils.kvstore import kvstore_dependencies


def available_providers() -> list[ProviderSpec]:
    return [
        remote_provider_spec(
            api=Api.files,
            adapter=AdapterSpec(
                adapter_type="s3",
                pip_packages=["aioboto3"] + kvstore_dependencies(),
                module="llama_stack.providers.remote.files.object.s3",
                config_class="llama_stack.providers.remote.files.object.s3.config.S3FilesImplConfig",
                provider_data_validator="llama_stack.providers.remote.files.object.s3.S3ProviderDataValidator",
            ),
        ),
    ]
