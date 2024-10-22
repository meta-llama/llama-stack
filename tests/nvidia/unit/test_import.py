# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.adapters.inference.nvidia import __all__


def test_import():
    assert set(__all__) == {"get_adapter_impl", "NVIDIAConfig"}
