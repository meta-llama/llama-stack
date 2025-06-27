# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.datatypes import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
)
from llama_stack.providers.utils.sqlstore.sqlstore import sql_store_pip_packages


def available_providers() -> list[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.files,
            provider_type="inline::localfs",
            # TODO: make this dynamic according to the sql store type
            pip_packages=sql_store_pip_packages,
            module="llama_stack.providers.inline.files.localfs",
            config_class="llama_stack.providers.inline.files.localfs.config.LocalfsFilesImplConfig",
            description="Local filesystem-based file storage provider for managing files and documents locally.",
        ),
    ]
