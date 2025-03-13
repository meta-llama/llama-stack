# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from pydantic import BaseModel

from llama_stack.distribution.distribution import get_provider_registry, providable_apis
from llama_stack.distribution.utils.dynamic import instantiate_class_type


def test_all_provider_configs_can_be_instantiated():
    """
    Test that all provider configs can be instantiated.
    This ensures that all config classes are correctly defined and can be instantiated without errors.
    """
    # Get all provider registries
    provider_registry = get_provider_registry()

    # Track any failures
    failures = []

    # For each API type
    for api in providable_apis():
        providers = provider_registry.get(api, {})

        # For each provider of this API type
        for provider_type, provider_spec in providers.items():
            try:
                # Get the config class
                config_class_name = provider_spec.config_class
                config_type = instantiate_class_type(config_class_name)

                assert issubclass(config_type, BaseModel)

            except Exception as e:
                failures.append(f"Failed to instantiate {provider_type} config: {str(e)}")

    # Report all failures at once
    if failures:
        pytest.fail("\n".join(failures))
