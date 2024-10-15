# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Union

from .config import MetaReferenceInferenceConfig, MetaReferenceQuantizedInferenceConfig


async def get_provider_impl(
    config: Union[MetaReferenceInferenceConfig, MetaReferenceQuantizedInferenceConfig],
    _deps,
):
    from .inference import MetaReferenceInferenceImpl

    impl = MetaReferenceInferenceImpl(config)
    await impl.initialize()
    return impl
