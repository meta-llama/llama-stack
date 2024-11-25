# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import importlib.util
import json
import tempfile

from typing import Any, Dict, List, Optional

from llama_stack.distribution.datatypes import *  # noqa: F403
from llama_stack.distribution.build import print_pip_install_help
from llama_stack.distribution.configure import parse_and_maybe_upgrade_config
from llama_stack.distribution.distribution import get_provider_registry
from llama_stack.distribution.request_headers import set_request_provider_data
from llama_stack.distribution.resolver import resolve_remote_stack_impls
from llama_stack.distribution.stack import construct_stack
from llama_stack.providers.utils.kvstore import SqliteKVStoreConfig


class TestStack(BaseModel):
    impls: Dict[Api, Any]
    run_config: StackRunConfig


async def construct_stack_for_test(
    apis: List[Api],
    providers: Dict[str, List[Provider]],
    provider_data: Optional[Dict[str, Any]] = None,
    models: Optional[List[ModelInput]] = None,
    shields: Optional[List[ShieldInput]] = None,
    memory_banks: Optional[List[MemoryBankInput]] = None,
    datasets: Optional[List[DatasetInput]] = None,
    scoring_fns: Optional[List[ScoringFnInput]] = None,
    eval_tasks: Optional[List[EvalTaskInput]] = None,
) -> TestStack:
    sqlite_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    run_config = dict(
        image_name="test-fixture",
        apis=apis,
        providers=providers,
        metadata_store=SqliteKVStoreConfig(db_path=sqlite_file.name),
        models=models or [],
        shields=shields or [],
        memory_banks=memory_banks or [],
        datasets=datasets or [],
        scoring_fns=scoring_fns or [],
        eval_tasks=eval_tasks or [],
    )
    run_config = parse_and_maybe_upgrade_config(run_config)
    try:
        remote_config = remote_provider_config(run_config)
        if not remote_config:
            # Here we create instance of registry with optional fake provider
            provider_registry = setup_provider_registry_for_test(
                run_config, get_provider_registry()
            )
            impls = await construct_stack(run_config, provider_registry)
        else:
            # we don't register resources for a remote stack as part of the fixture setup
            # because the stack is already "up". if a test needs to register resources, it
            # can do so manually always.

            impls = await resolve_remote_stack_impls(remote_config, run_config.apis)

        test_stack = TestStack(impls=impls, run_config=run_config)
    except ModuleNotFoundError as e:
        print_pip_install_help(providers)
        raise e

    if provider_data:
        set_request_provider_data(
            {"X-LlamaStack-ProviderData": json.dumps(provider_data)}
        )

    return test_stack


def setup_provider_registry_for_test(
    run_config: StackRunConfig, provider_registry: Dict[Api, Dict[str, ProviderSpec]]
) -> Dict[Api, Dict[str, ProviderSpec]]:
    provider_registry = get_provider_registry()
    for api_name, providers in run_config.providers.items():
        for provider in providers:
            if provider.provider_type == "test::fake":
                # Check if the fake provider module exists for the API that is trying
                # to use test::fake provider_type
                provider_fake_module_name = (
                    f"llama_stack.providers.tests.{api_name}.fakes"
                )
                provider_fake_module_spec = importlib.util.find_spec(
                    provider_fake_module_name
                )
                if provider_fake_module_spec is None:
                    raise ValueError(
                        f"Fake provider module {provider_fake_module_name} does not exist. "
                        f"The module must be defined inside the providers/tests/{api_name}/fakes.py file."
                    )

                # Import the module so we can validate that the config class exists
                provider_fake_module = importlib.import_module(
                    provider_fake_module_name
                )

                # Check if the fake provider config class exists
                # The class name is derived from the provider type e.g.
                #   provider_id: "example_provider" -> class_name: "ExampleProviderConfig"
                provider_fake_config_class_name = (
                    f"{provider.provider_id}_config".title().replace("_", "")
                )
                if not hasattr(provider_fake_module, provider_fake_config_class_name):
                    raise ValueError(
                        f"Fake provider config class {provider_fake_config_class_name} "
                        f"does not exist in module {provider_fake_module_name}. "
                        f"The config class must be defined inside the providers/tests/{api_name}/fakes.py file."
                    )

                provider_fake_config_class_path = f"llama_stack.providers.tests.{api_name}.fakes.{provider_fake_config_class_name}"

                api = getattr(Api, api_name)
                fake_api_spec = InlineProviderSpec(
                    api=api,
                    provider_type="test::fake",
                    pip_packages=[],
                    module=provider_fake_module_name,
                    config_class=provider_fake_config_class_path,
                )
                provider_registry[api].update({"test::fake": fake_api_spec})

    return provider_registry


def remote_provider_config(
    run_config: StackRunConfig,
) -> Optional[RemoteProviderConfig]:
    remote_config = None
    has_non_remote = False
    for api_providers in run_config.providers.values():
        for provider in api_providers:
            if provider.provider_type == "test::remote":
                remote_config = RemoteProviderConfig(**provider.config)
            else:
                has_non_remote = True

    if remote_config:
        assert not has_non_remote, "Remote stack cannot have non-remote providers"

    return remote_config
