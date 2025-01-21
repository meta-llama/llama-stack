# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel

from .config import CentMLImplConfig


class CentMLProviderDataValidator(BaseModel):
    centml_api_key: str


async def get_adapter_impl(config: CentMLImplConfig, _deps):
    """
    Factory function to construct and initialize the CentML adapter.

    :param config: Instance of CentMLImplConfig, containing `url`, `api_key`, etc.
    :param _deps: Additional dependencies provided by llama-stack (unused here).
    """
    from .centml import CentMLInferenceAdapter

    # Ensure the provided config is indeed a CentMLImplConfig
    assert isinstance(config, CentMLImplConfig), (
        f"Unexpected config type: {type(config)}"
    )

    # Instantiate and initialize the adapter
    adapter = CentMLInferenceAdapter(config)
    await adapter.initialize()
    return adapter
