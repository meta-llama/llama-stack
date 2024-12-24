# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import tempfile
from typing import Any, Dict, List, Optional

from llama_stack.distribution.build import print_pip_install_help
from llama_stack.distribution.configure import parse_and_maybe_upgrade_config
from llama_stack.distribution.datatypes import *  # noqa: F403
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
    tool_groups: Optional[List[ToolGroupInput]] = None,
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
        tool_groups=tool_groups or [],
    )
    run_config = parse_and_maybe_upgrade_config(run_config)
    try:
        remote_config = remote_provider_config(run_config)
        if not remote_config:
            # TODO: add to provider registry by creating interesting mocks or fakes
            impls = await construct_stack(run_config, get_provider_registry())
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
