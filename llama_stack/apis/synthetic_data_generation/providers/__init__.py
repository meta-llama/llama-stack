# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from typing import cast

from llama_stack.apis.synthetic_data_generation import SyntheticDataGeneration
from llama_stack.providers.utils.resolver import get_provider_impl as _get_provider_impl


def get_provider_impl() -> SyntheticDataGeneration:
    return cast(SyntheticDataGeneration, _get_provider_impl(SyntheticDataGeneration))
