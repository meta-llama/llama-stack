# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.synthetic_data_generation import SyntheticDataGeneration

SYNTHETIC_DATA_GENERATION_PROVIDERS: dict[str, SyntheticDataGeneration] = {}


def get_provider(name: str = "meta_synthetic_data_kit") -> SyntheticDataGeneration:
    raise NotImplementedError(f"No provider registered yet for synthetic_data_generation (requested: {name})")
