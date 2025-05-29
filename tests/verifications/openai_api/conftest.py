# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from tests.verifications.openai_api.fixtures.fixtures import _load_all_verification_configs


def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests based on the selected provider and config."""
    if "model" in metafunc.fixturenames:
        model = metafunc.config.getoption("model")
        if model:
            metafunc.parametrize("model", [model])
            return

        provider = metafunc.config.getoption("provider")
        if not provider:
            print("Warning: --provider not specified. Skipping model parametrization.")
            metafunc.parametrize("model", [])
            return

        try:
            config_data = _load_all_verification_configs()
        except (OSError, FileNotFoundError) as e:
            print(f"ERROR loading verification configs: {e}")
            config_data = {"providers": {}}

        provider_config = config_data.get("providers", {}).get(provider)
        if provider_config:
            models = provider_config.get("models", [])
            if models:
                metafunc.parametrize("model", models)
            else:
                print(f"Warning: No models found for provider '{provider}' in config.")
                metafunc.parametrize("model", [])  # Parametrize empty if no models found
        else:
            print(f"Warning: Provider '{provider}' not found in config. No models parametrized.")
            metafunc.parametrize("model", [])  # Parametrize empty if provider not found
