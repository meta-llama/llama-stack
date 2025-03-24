# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.apis.models.models import ModelType
from llama_stack.distribution.datatypes import (
    ModelInput,
    Provider,
    ShieldInput,
    ToolGroupInput,
)
from llama_stack.providers.inline.inference.sentence_transformers import (
    SentenceTransformersInferenceConfig,
)
from llama_stack.providers.inline.vector_io.sqlite_vec.config import (
    SQLiteVectorIOConfig,
)
from llama_stack.providers.remote.inference.fireworks.config import FireworksImplConfig
from llama_stack.providers.remote.inference.fireworks.models import MODEL_ENTRIES
from llama_stack.templates.template import (
    DistributionTemplate,
    RunConfigSettings,
    get_model_registry,
)


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": ["remote::fireworks", "inline::sentence-transformers"],
        "vector_io": ["inline::sqlite-vec", "remote::chromadb", "remote::pgvector"],
        "safety": ["inline::llama-guard"],
        "agents": ["inline::meta-reference"],
        "telemetry": ["inline::meta-reference"],
        "eval": ["inline::meta-reference"],
        "datasetio": ["remote::huggingface", "inline::localfs"],
        "scoring": ["inline::basic", "inline::llm-as-judge", "inline::braintrust"],
        "tool_runtime": [
            "remote::brave-search",
            "remote::tavily-search",
            "inline::code-interpreter",
            "inline::rag-runtime",
            "remote::model-context-protocol",
        ],
    }
    name = "ci-tests"
    inference_provider = Provider(
        provider_id="fireworks",
        provider_type="remote::fireworks",
        config=FireworksImplConfig.sample_run_config(),
    )
    vector_io_provider = Provider(
        provider_id="sqlite-vec",
        provider_type="inline::sqlite-vec",
        config=SQLiteVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
    )
    embedding_provider = Provider(
        provider_id="sentence-transformers",
        provider_type="inline::sentence-transformers",
        config=SentenceTransformersInferenceConfig.sample_run_config(),
    )

    default_tool_groups = [
        ToolGroupInput(
            toolgroup_id="builtin::websearch",
            provider_id="tavily-search",
        ),
        ToolGroupInput(
            toolgroup_id="builtin::rag",
            provider_id="rag-runtime",
        ),
        ToolGroupInput(
            toolgroup_id="builtin::code_interpreter",
            provider_id="code-interpreter",
        ),
    ]
    available_models = {
        "fireworks": MODEL_ENTRIES,
    }
    default_models = get_model_registry(available_models)
    embedding_model = ModelInput(
        model_id="all-MiniLM-L6-v2",
        provider_id="sentence-transformers",
        model_type=ModelType.embedding,
        metadata={
            "embedding_dimension": 384,
        },
    )

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Distribution for running e2e tests in CI",
        container_image=None,
        template_path=None,
        providers=providers,
        available_models_by_provider=available_models,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider, embedding_provider],
                    "vector_io": [vector_io_provider],
                },
                default_models=default_models + [embedding_model],
                default_tool_groups=default_tool_groups,
                default_shields=[ShieldInput(shield_id="meta-llama/Llama-Guard-3-8B")],
            ),
        },
        run_config_env_vars={
            "LLAMA_STACK_PORT": (
                "8321",
                "Port for the Llama Stack distribution server",
            ),
            "FIREWORKS_API_KEY": (
                "",
                "Fireworks API Key",
            ),
        },
    )
