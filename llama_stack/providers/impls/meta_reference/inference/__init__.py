# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import MetaReferenceImplConfig  # noqa


async def get_provider_impl(config: MetaReferenceImplConfig, _deps):
    from .inference import MetaReferenceInferenceImpl

    assert isinstance(
        config, MetaReferenceImplConfig
    ), f"Unexpected config type: {type(config)}"

    impl = MetaReferenceInferenceImpl(config)
    await impl.initialize()
    return impl
