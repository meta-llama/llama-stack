# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import textwrap

from typing import Any

from llama_models.sku_list import (
    llama3_1_family,
    llama3_2_family,
    llama3_family,
    resolve_model,
    safety_models,
)

from llama_stack.distribution.datatypes import *  # noqa: F403
from prompt_toolkit import prompt
from prompt_toolkit.validation import Validator
from termcolor import cprint

from llama_stack.distribution.distribution import (
    builtin_automatically_routed_apis,
    get_provider_registry,
)
from llama_stack.distribution.utils.dynamic import instantiate_class_type
from llama_stack.distribution.utils.prompt_for_config import prompt_for_config


from llama_stack.apis.models import *  # noqa: F403
from llama_stack.apis.shields import *  # noqa: F403
from llama_stack.apis.memory_banks import *  # noqa: F403


ALLOWED_MODELS = (
    llama3_family() + llama3_1_family() + llama3_2_family() + safety_models()
)


def configure_single_provider(
    registry: Dict[str, ProviderSpec], provider: Provider
) -> Provider:
    provider_spec = registry[provider.provider_type]
    config_type = instantiate_class_type(provider_spec.config_class)
    try:
        if provider.config:
            existing = config_type(**provider.config)
        else:
            existing = None
    except Exception:
        existing = None

    cfg = prompt_for_config(config_type, existing)
    return Provider(
        provider_id=provider.provider_id,
        provider_type=provider.provider_type,
        config=cfg.dict(),
    )


def configure_api_providers(
    config: StackRunConfig, build_spec: DistributionSpec
) -> StackRunConfig:
    is_nux = len(config.providers) == 0

    if is_nux:
        print(
            textwrap.dedent(
                """
        Llama Stack is composed of several APIs working together. For each API served by the Stack,
        we need to configure the providers (implementations) you want to use for these APIs.
"""
            )
        )

    provider_registry = get_provider_registry()
    builtin_apis = [a.routing_table_api for a in builtin_automatically_routed_apis()]

    if config.apis:
        apis_to_serve = config.apis
    else:
        apis_to_serve = [a.value for a in Api if a not in (Api.telemetry, Api.inspect)]

    for api_str in apis_to_serve:
        api = Api(api_str)
        if api in builtin_apis:
            continue
        if api not in provider_registry:
            raise ValueError(f"Unknown API `{api_str}`")

        existing_providers = config.providers.get(api_str, [])
        if existing_providers:
            cprint(
                f"Re-configuring existing providers for API `{api_str}`...",
                "green",
                attrs=["bold"],
            )
            updated_providers = []
            for p in existing_providers:
                print(f"> Configuring provider `({p.provider_type})`")
                updated_providers.append(
                    configure_single_provider(provider_registry[api], p)
                )
                print("")
        else:
            # we are newly configuring this API
            plist = build_spec.providers.get(api_str, [])
            plist = plist if isinstance(plist, list) else [plist]

            if not plist:
                raise ValueError(f"No provider configured for API {api_str}?")

            cprint(f"Configuring API `{api_str}`...", "green", attrs=["bold"])
            updated_providers = []
            for i, provider_type in enumerate(plist):
                print(f"> Configuring provider `({provider_type})`")
                updated_providers.append(
                    configure_single_provider(
                        provider_registry[api],
                        Provider(
                            provider_id=(
                                f"{provider_type}-{i:02d}"
                                if len(plist) > 1
                                else provider_type
                            ),
                            provider_type=provider_type,
                            config={},
                        ),
                    )
                )
                print("")

        config.providers[api_str] = updated_providers

    if is_nux:
        print(
            textwrap.dedent(
                """
        =========================================================================================
        Now let's configure the `objects` you will be serving via the stack. These are:

        - Models: the Llama model SKUs you expect to inference (e.g., Llama3.2-1B-Instruct)
        - Shields: the safety models you expect to use for safety (e.g., Llama-Guard-3-1B)
        - Memory Banks: the memory banks you expect to use for memory (e.g., Vector stores)

        This wizard will guide you through setting up one of each of these objects. You can
        always add more later by editing the run.yaml file.
        """
            )
        )

    object_types = {
        "models": (ModelDef, configure_models, "inference"),
        "shields": (ShieldDef, configure_shields, "safety"),
        "memory_banks": (MemoryBankDef, configure_memory_banks, "memory"),
    }
    safety_providers = config.providers.get("safety", [])

    for otype, (odef, config_method, api_str) in object_types.items():
        existing_objects = getattr(config, otype)

        if existing_objects:
            cprint(
                f"{len(existing_objects)} {otype} exist. Skipping...",
                "blue",
                attrs=["bold"],
            )
            updated_objects = existing_objects
        else:
            providers = config.providers.get(api_str, [])
            if not providers:
                updated_objects = []
            else:
                # we are newly configuring this API
                cprint(f"Configuring `{otype}`...", "blue", attrs=["bold"])
                updated_objects = config_method(
                    config.providers[api_str], safety_providers
                )

        setattr(config, otype, updated_objects)
        print("")

    return config


def get_llama_guard_model(safety_providers: List[Provider]) -> Optional[str]:
    if not safety_providers:
        return None

    provider = safety_providers[0]
    assert provider.provider_type == "meta-reference"

    cfg = provider.config["llama_guard_shield"]
    if not cfg:
        return None
    return cfg["model"]


def configure_models(
    providers: List[Provider], safety_providers: List[Provider]
) -> List[ModelDef]:
    model = prompt(
        "> Please enter the model you want to serve: ",
        default="Llama3.2-1B-Instruct",
        validator=Validator.from_callable(
            lambda x: resolve_model(x) is not None,
            error_message="Model must be: {}".format(
                [x.descriptor() for x in ALLOWED_MODELS]
            ),
        ),
    )
    model = ModelDef(
        identifier=model,
        llama_model=model,
        provider_id=providers[0].provider_id,
    )

    ret = [model]
    if llama_guard := get_llama_guard_model(safety_providers):
        ret.append(
            ModelDef(
                identifier=llama_guard,
                llama_model=llama_guard,
                provider_id=providers[0].provider_id,
            )
        )

    return ret


def configure_shields(
    providers: List[Provider], safety_providers: List[Provider]
) -> List[ShieldDef]:
    if get_llama_guard_model(safety_providers):
        return [
            ShieldDef(
                identifier="llama_guard",
                type="llama_guard",
                provider_id=providers[0].provider_id,
                params={},
            )
        ]

    return []


def configure_memory_banks(
    providers: List[Provider], safety_providers: List[Provider]
) -> List[MemoryBankDef]:
    bank_name = prompt(
        "> Please enter a name for your memory bank: ",
        default="my-memory-bank",
    )

    return [
        VectorMemoryBankDef(
            identifier=bank_name,
            provider_id=providers[0].provider_id,
            embedding_model="all-MiniLM-L6-v2",
            chunk_size_in_tokens=512,
        )
    ]


def upgrade_from_routing_table_to_registry(
    config_dict: Dict[str, Any],
) -> Dict[str, Any]:
    def get_providers(entries):
        return [
            Provider(
                provider_id=(
                    f"{entry['provider_type']}-{i:02d}"
                    if len(entries) > 1
                    else entry["provider_type"]
                ),
                provider_type=entry["provider_type"],
                config=entry["config"],
            )
            for i, entry in enumerate(entries)
        ]

    providers_by_api = {}
    models = []
    shields = []
    memory_banks = []

    routing_table = config_dict.get("routing_table", {})
    for api_str, entries in routing_table.items():
        providers = get_providers(entries)
        providers_by_api[api_str] = providers

        if api_str == "inference":
            for entry, provider in zip(entries, providers):
                key = entry["routing_key"]
                keys = key if isinstance(key, list) else [key]
                for key in keys:
                    models.append(
                        ModelDef(
                            identifier=key,
                            provider_id=provider.provider_id,
                            llama_model=key,
                        )
                    )
        elif api_str == "safety":
            for entry, provider in zip(entries, providers):
                key = entry["routing_key"]
                keys = key if isinstance(key, list) else [key]
                for key in keys:
                    shields.append(
                        ShieldDef(
                            identifier=key,
                            type=ShieldType.llama_guard.value,
                            provider_id=provider.provider_id,
                        )
                    )
        elif api_str == "memory":
            for entry, provider in zip(entries, providers):
                key = entry["routing_key"]
                keys = key if isinstance(key, list) else [key]
                for key in keys:
                    # we currently only support Vector memory banks so this is OK
                    memory_banks.append(
                        VectorMemoryBankDef(
                            identifier=key,
                            provider_id=provider.provider_id,
                            embedding_model="all-MiniLM-L6-v2",
                            chunk_size_in_tokens=512,
                        )
                    )
    config_dict["models"] = models
    config_dict["shields"] = shields
    config_dict["memory_banks"] = memory_banks

    provider_map = config_dict.get("api_providers", config_dict.get("provider_map", {}))
    if provider_map:
        for api_str, provider in provider_map.items():
            if isinstance(provider, dict):
                providers_by_api[api_str] = [
                    Provider(
                        provider_id=f"{provider['provider_type']}",
                        provider_type=provider["provider_type"],
                        config=provider["config"],
                    )
                ]

    config_dict["providers"] = providers_by_api

    config_dict.pop("routing_table", None)
    config_dict.pop("api_providers", None)
    config_dict.pop("provider_map", None)

    config_dict["apis"] = config_dict["apis_to_serve"]
    config_dict.pop("apis_to_serve", None)

    return config_dict


def parse_and_maybe_upgrade_config(config_dict: Dict[str, Any]) -> StackRunConfig:
    version = config_dict.get("version", None)
    if version == LLAMA_STACK_RUN_CONFIG_VERSION:
        return StackRunConfig(**config_dict)

    if "models" not in config_dict:
        print("Upgrading config...")
        config_dict = upgrade_from_routing_table_to_registry(config_dict)

    config_dict["version"] = LLAMA_STACK_RUN_CONFIG_VERSION
    config_dict["built_at"] = datetime.now().isoformat()

    return StackRunConfig(**config_dict)
