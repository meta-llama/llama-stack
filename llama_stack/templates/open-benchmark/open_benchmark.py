# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict, List, Tuple

from llama_stack.apis.datasets import DatasetPurpose, URIDataSource
from llama_stack.apis.models.models import ModelType
from llama_stack.distribution.datatypes import (
    BenchmarkInput,
    DatasetInput,
    ModelInput,
    Provider,
    ShieldInput,
    ToolGroupInput,
)
from llama_stack.providers.inline.vector_io.sqlite_vec.config import (
    SQLiteVectorIOConfig,
)
from llama_stack.providers.remote.inference.anthropic.config import AnthropicConfig
from llama_stack.providers.remote.inference.gemini.config import GeminiConfig
from llama_stack.providers.remote.inference.groq.config import GroqConfig
from llama_stack.providers.remote.inference.openai.config import OpenAIConfig
from llama_stack.providers.remote.inference.together.config import TogetherImplConfig
from llama_stack.providers.remote.vector_io.chroma.config import ChromaVectorIOConfig
from llama_stack.providers.remote.vector_io.pgvector.config import (
    PGVectorVectorIOConfig,
)
from llama_stack.providers.utils.inference.model_registry import ProviderModelEntry
from llama_stack.templates.template import (
    DistributionTemplate,
    RunConfigSettings,
    get_model_registry,
)


def get_inference_providers() -> Tuple[List[Provider], Dict[str, List[ProviderModelEntry]]]:
    # in this template, we allow each API key to be optional
    providers = [
        (
            "openai",
            [
                ProviderModelEntry(
                    provider_model_id="openai/gpt-4o",
                    model_type=ModelType.llm,
                )
            ],
            OpenAIConfig.sample_run_config(api_key="${env.OPENAI_API_KEY:}"),
        ),
        (
            "anthropic",
            [
                ProviderModelEntry(
                    provider_model_id="anthropic/claude-3-5-sonnet-latest",
                    model_type=ModelType.llm,
                )
            ],
            AnthropicConfig.sample_run_config(api_key="${env.ANTHROPIC_API_KEY:}"),
        ),
        (
            "gemini",
            [
                ProviderModelEntry(
                    provider_model_id="gemini/gemini-1.5-flash",
                    model_type=ModelType.llm,
                )
            ],
            GeminiConfig.sample_run_config(api_key="${env.GEMINI_API_KEY:}"),
        ),
        (
            "groq",
            [],
            GroqConfig.sample_run_config(api_key="${env.GROQ_API_KEY:}"),
        ),
        (
            "together",
            [],
            TogetherImplConfig.sample_run_config(api_key="${env.TOGETHER_API_KEY:}"),
        ),
    ]
    inference_providers = []
    available_models = {}
    for provider_id, model_entries, config in providers:
        inference_providers.append(
            Provider(
                provider_id=provider_id,
                provider_type=f"remote::{provider_id}",
                config=config,
            )
        )
        available_models[provider_id] = model_entries
    return inference_providers, available_models


def get_distribution_template() -> DistributionTemplate:
    inference_providers, available_models = get_inference_providers()
    providers = {
        "inference": [p.provider_type for p in inference_providers],
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
    name = "open-benchmark"

    vector_io_providers = [
        Provider(
            provider_id="sqlite-vec",
            provider_type="inline::sqlite-vec",
            config=SQLiteVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
        ),
        Provider(
            provider_id="${env.ENABLE_CHROMADB+chromadb}",
            provider_type="remote::chromadb",
            config=ChromaVectorIOConfig.sample_run_config(url="${env.CHROMADB_URL:}"),
        ),
        Provider(
            provider_id="${env.ENABLE_PGVECTOR+pgvector}",
            provider_type="remote::pgvector",
            config=PGVectorVectorIOConfig.sample_run_config(
                db="${env.PGVECTOR_DB:}",
                user="${env.PGVECTOR_USER:}",
                password="${env.PGVECTOR_PASSWORD:}",
            ),
        ),
    ]

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

    default_models = get_model_registry(available_models) + [
        ModelInput(
            model_id="meta-llama/Llama-3.3-70B-Instruct",
            provider_id="groq",
            provider_model_id="groq/llama-3.3-70b-versatile",
            model_type=ModelType.llm,
        ),
        ModelInput(
            model_id="meta-llama/Llama-3.1-405B-Instruct",
            provider_id="together",
            provider_model_id="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            model_type=ModelType.llm,
        ),
    ]

    default_datasets = [
        DatasetInput(
            dataset_id="simpleqa",
            purpose=DatasetPurpose.eval_messages_answer,
            source=URIDataSource(
                uri="huggingface://datasets/llamastack/simpleqa?split=train",
            ),
        ),
        DatasetInput(
            dataset_id="mmlu_cot",
            purpose=DatasetPurpose.eval_messages_answer,
            source=URIDataSource(
                uri="huggingface://datasets/llamastack/mmlu_cot?split=test&name=all",
            ),
        ),
        DatasetInput(
            dataset_id="gpqa_cot",
            purpose=DatasetPurpose.eval_messages_answer,
            source=URIDataSource(
                uri="huggingface://datasets/llamastack/gpqa_0shot_cot?split=test&name=gpqa_main",
            ),
        ),
        DatasetInput(
            dataset_id="gpqa_cot_diamond",
            purpose=DatasetPurpose.eval_messages_answer,
            source=URIDataSource(
                uri="huggingface://datasets/llamastack/gpqa_0shot_cot?split=test&name=gpqa_diamond",
            ),
        ),
        DatasetInput(
            dataset_id="gpqa",
            purpose=DatasetPurpose.eval_messages_answer,
            source=URIDataSource(
                uri="huggingface://datasets/llamastack/gpqa_0shot?split=test&name=gpqa_main",
            ),
        ),
        DatasetInput(
            dataset_id="math_500",
            purpose=DatasetPurpose.eval_messages_answer,
            source=URIDataSource(
                uri="huggingface://datasets/llamastack/math_500?split=test",
            ),
        ),
        DatasetInput(
            dataset_id="bfcl",
            purpose=DatasetPurpose.eval_messages_answer,
            source=URIDataSource(
                uri="huggingface://datasets/llamastack/bfcl_v3?split=train",
            ),
        ),
        DatasetInput(
            dataset_id="docvqa",
            purpose=DatasetPurpose.eval_messages_answer,
            source=URIDataSource(
                uri="huggingface://datasets/llamastack/docvqa?split=val",
            ),
        ),
        DatasetInput(
            dataset_id="MMMU",
            purpose=DatasetPurpose.eval_messages_answer,
            source=URIDataSource(
                uri="huggingface://datasets/llamastack/mmmu_v3?split=validation",
            ),
        ),
        DatasetInput(
            dataset_id="MMMU_Pro_standard",
            purpose=DatasetPurpose.eval_messages_answer,
            source=URIDataSource(
                uri="huggingface://datasets/llamastack/MMMU_Pro?name=standard%20(10%20options)&split=test",
            ),
        ),
        DatasetInput(
            dataset_id="MMMU_Pro_vision",
            purpose=DatasetPurpose.eval_messages_answer,
            source=URIDataSource(
                uri="huggingface://datasets/llamastack/MMMU_Pro?name=vision&split=test",
            ),
        ),
        DatasetInput(
            dataset_id="ai2d",
            purpose=DatasetPurpose.eval_messages_answer,
            source=URIDataSource(
                uri="huggingface://datasets/llamastack/ai2d?split=test",
            ),
        ),
    ]

    default_benchmarks = [
        BenchmarkInput(
            benchmark_id="meta-reference-simpleqa",
            dataset_id="simpleqa",
            scoring_functions=["llm-as-judge::405b-simpleqa"],
        ),
        BenchmarkInput(
            benchmark_id="meta-reference-mmlu-cot",
            dataset_id="mmlu_cot",
            scoring_functions=["basic::regex_parser_multiple_choice_answer"],
        ),
        BenchmarkInput(
            benchmark_id="meta-reference-gpqa-cot",
            dataset_id="gpqa_cot",
            scoring_functions=["basic::regex_parser_multiple_choice_answer"],
        ),
        BenchmarkInput(
            benchmark_id="meta-reference-gpqa-cot-diamond",
            dataset_id="gpqa_cot_diamond",
            scoring_functions=["basic::regex_parser_multiple_choice_answer"],
        ),
        BenchmarkInput(
            benchmark_id="meta-reference-gpqa",
            dataset_id="gpqa",
            scoring_functions=["basic::regex_parser_multiple_choice_answer"],
        ),
        BenchmarkInput(
            benchmark_id="meta-reference-math-500",
            dataset_id="math_500",
            scoring_functions=["basic::regex_parser_math_response"],
        ),
        BenchmarkInput(
            benchmark_id="meta-reference-bfcl",
            dataset_id="bfcl",
            scoring_functions=["basic::bfcl"],
        ),
        BenchmarkInput(
            benchmark_id="meta-reference-docvqa",
            dataset_id="docvqa",
            scoring_functions=["basic::docvqa"],
        ),
        BenchmarkInput(
            benchmark_id="meta-reference-MMMU",
            dataset_id="MMMU",
            scoring_functions=["basic::regex_parser_multiple_choice_answer"],
        ),
        BenchmarkInput(
            benchmark_id="meta-reference-MMMU_Pro_standard",
            dataset_id="MMMU_Pro_standard",
            scoring_functions=["basic::regex_parser_multiple_choice_answer"],
        ),
        BenchmarkInput(
            benchmark_id="meta-reference-MMMU_Pro_vision",
            dataset_id="MMMU_Pro_vision",
            scoring_functions=["basic::regex_parser_multiple_choice_answer"],
        ),
        BenchmarkInput(
            benchmark_id="meta-reference-ai2d",
            dataset_id="ai2d",
            scoring_functions=["basic::regex_parser_multiple_choice_answer"],
        ),
    ]
    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Distribution for running open benchmarks",
        container_image=None,
        template_path=None,
        providers=providers,
        available_models_by_provider=available_models,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": inference_providers,
                    "vector_io": vector_io_providers,
                },
                default_models=default_models,
                default_tool_groups=default_tool_groups,
                default_shields=[ShieldInput(shield_id="meta-llama/Llama-Guard-3-8B")],
                default_datasets=default_datasets,
                default_benchmarks=default_benchmarks,
            ),
        },
        run_config_env_vars={
            "LLAMA_STACK_PORT": (
                "5001",
                "Port for the Llama Stack distribution server",
            ),
            "TOGETHER_API_KEY": (
                "",
                "Together API Key",
            ),
            "OPENAI_API_KEY": (
                "",
                "OpenAI API Key",
            ),
            "GEMINI_API_KEY": (
                "",
                "Gemini API Key",
            ),
            "ANTHROPIC_API_KEY": (
                "",
                "Anthropic API Key",
            ),
            "GROQ_API_KEY": (
                "",
                "Groq API Key",
            ),
        },
    )
