# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.models.models import ModelType
from llama_stack.providers.utils.inference.model_registry import (
    ProviderModelEntry,
)

model_entries = [
    ProviderModelEntry(
        provider_model_id="ibm-granite/granite-3.3-8b-instruct",
        aliases=["ibm-granite/granite-3.3-8b-instruct"],
        model_type=ModelType.llm,
    ),
    ProviderModelEntry(
        provider_model_id="ibm-granite/granite-3.3-8b-instruct",
        aliases=["ibm-granite/granite-3.3-8b-instruct"],
        model_type=ModelType.llm,
    ),
]
