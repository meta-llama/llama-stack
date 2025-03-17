# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import jinja2
import yaml
from pydantic import BaseModel, Field

from llama_stack.apis.datasets import DatasetPurpose
from llama_stack.apis.models.models import ModelType
from llama_stack.distribution.datatypes import (
    Api,
    BenchmarkInput,
    BuildConfig,
    DatasetInput,
    DistributionSpec,
    ModelInput,
    Provider,
    ShieldInput,
    StackRunConfig,
    ToolGroupInput,
)
from llama_stack.distribution.distribution import get_provider_registry
from llama_stack.distribution.utils.dynamic import instantiate_class_type
from llama_stack.providers.utils.inference.model_registry import ProviderModelEntry
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig


def get_model_registry(
    available_models: Dict[str, List[ProviderModelEntry]],
) -> List[ModelInput]:
    models = []
    for provider_id, entries in available_models.items():
        for entry in entries:
            ids = [entry.provider_model_id] + entry.aliases
            for model_id in ids:
                models.append(
                    ModelInput(
                        model_id=model_id,
                        provider_model_id=entry.provider_model_id,
                        provider_id=provider_id,
                        model_type=entry.model_type,
                        metadata=entry.metadata,
                    )
                )
    return models


class DefaultModel(BaseModel):
    model_id: str
    doc_string: str


class RunConfigSettings(BaseModel):
    provider_overrides: Dict[str, List[Provider]] = Field(default_factory=dict)
    default_models: Optional[List[ModelInput]] = None
    default_shields: Optional[List[ShieldInput]] = None
    default_tool_groups: Optional[List[ToolGroupInput]] = None
    default_datasets: Optional[List[DatasetInput]] = None
    default_benchmarks: Optional[List[BenchmarkInput]] = None

    def run_config(
        self,
        name: str,
        providers: Dict[str, List[str]],
        container_image: Optional[str] = None,
    ) -> StackRunConfig:
        provider_registry = get_provider_registry()

        provider_configs = {}
        for api_str, provider_types in providers.items():
            if api_providers := self.provider_overrides.get(api_str):
                provider_configs[api_str] = api_providers
                continue

            provider_configs[api_str] = []
            for provider_type in provider_types:
                provider_id = provider_type.split("::")[-1]

                api = Api(api_str)
                if provider_type not in provider_registry[api]:
                    raise ValueError(f"Unknown provider type: {provider_type} for API: {api_str}")

                config_class = provider_registry[api][provider_type].config_class
                assert config_class is not None, (
                    f"No config class for provider type: {provider_type} for API: {api_str}"
                )

                config_class = instantiate_class_type(config_class)
                if hasattr(config_class, "sample_run_config"):
                    config = config_class.sample_run_config(__distro_dir__=f"~/.llama/distributions/{name}")
                else:
                    config = {}

                provider_configs[api_str].append(
                    Provider(
                        provider_id=provider_id,
                        provider_type=provider_type,
                        config=config,
                    )
                )

        # Get unique set of APIs from providers
        apis = sorted(providers.keys())

        return StackRunConfig(
            image_name=name,
            container_image=container_image,
            apis=apis,
            providers=provider_configs,
            metadata_store=SqliteKVStoreConfig.sample_run_config(
                __distro_dir__=f"~/.llama/distributions/{name}",
                db_name="registry.db",
            ),
            models=self.default_models or [],
            shields=self.default_shields or [],
            tool_groups=self.default_tool_groups or [],
            datasets=self.default_datasets or [],
            benchmarks=self.default_benchmarks or [],
        )


class DistributionTemplate(BaseModel):
    """
    Represents a Llama Stack distribution instance that can generate configuration
    and documentation files.
    """

    name: str
    description: str
    distro_type: Literal["self_hosted", "remote_hosted", "ondevice"]

    providers: Dict[str, List[str]]
    run_configs: Dict[str, RunConfigSettings]
    template_path: Optional[Path] = None

    # Optional configuration
    run_config_env_vars: Optional[Dict[str, Tuple[str, str]]] = None
    container_image: Optional[str] = None

    available_models_by_provider: Optional[Dict[str, List[ProviderModelEntry]]] = None

    def build_config(self) -> BuildConfig:
        return BuildConfig(
            name=self.name,
            distribution_spec=DistributionSpec(
                description=self.description,
                container_image=self.container_image,
                providers=self.providers,
            ),
            image_type="conda",  # default to conda, can be overridden
        )

    def generate_markdown_docs(self) -> str:
        providers_table = "| API | Provider(s) |\n"
        providers_table += "|-----|-------------|\n"

        for api, providers in sorted(self.providers.items()):
            providers_str = ", ".join(f"`{p}`" for p in providers)
            providers_table += f"| {api} | {providers_str} |\n"

        template = self.template_path.read_text()
        comment = "<!-- This file was auto-generated by distro_codegen.py, please edit source -->\n"
        orphantext = "---\norphan: true\n---\n"

        if template.startswith(orphantext):
            template = template.replace(orphantext, orphantext + comment)
        else:
            template = comment + template

        # Render template with rich-generated table
        env = jinja2.Environment(
            trim_blocks=True,
            lstrip_blocks=True,
            # NOTE: autoescape is required to prevent XSS attacks
            autoescape=True,
        )
        template = env.from_string(template)

        default_models = []
        if self.available_models_by_provider:
            has_multiple_providers = len(self.available_models_by_provider.keys()) > 1
            for provider_id, model_entries in self.available_models_by_provider.items():
                for model_entry in model_entries:
                    doc_parts = []
                    if model_entry.aliases:
                        doc_parts.append(f"aliases: {', '.join(model_entry.aliases)}")
                    if has_multiple_providers:
                        doc_parts.append(f"provider: {provider_id}")

                    default_models.append(
                        DefaultModel(
                            model_id=model_entry.provider_model_id,
                            doc_string=(f"({' -- '.join(doc_parts)})" if doc_parts else ""),
                        )
                    )

        return template.render(
            name=self.name,
            description=self.description,
            providers=self.providers,
            providers_table=providers_table,
            run_config_env_vars=self.run_config_env_vars,
            default_models=default_models,
        )

    def save_distribution(self, yaml_output_dir: Path, doc_output_dir: Path) -> None:
        def enum_representer(dumper, data):
            return dumper.represent_scalar("tag:yaml.org,2002:str", data.value)

        # Register YAML representer for ModelType
        yaml.add_representer(ModelType, enum_representer)
        yaml.add_representer(DatasetPurpose, enum_representer)
        yaml.SafeDumper.add_representer(ModelType, enum_representer)
        yaml.SafeDumper.add_representer(DatasetPurpose, enum_representer)

        for output_dir in [yaml_output_dir, doc_output_dir]:
            output_dir.mkdir(parents=True, exist_ok=True)

        build_config = self.build_config()
        with open(yaml_output_dir / "build.yaml", "w") as f:
            yaml.safe_dump(
                build_config.model_dump(exclude_none=True),
                f,
                sort_keys=False,
            )

        for yaml_pth, settings in self.run_configs.items():
            run_config = settings.run_config(self.name, self.providers, self.container_image)
            with open(yaml_output_dir / yaml_pth, "w") as f:
                yaml.safe_dump(
                    run_config.model_dump(exclude_none=True),
                    f,
                    sort_keys=False,
                )

        if self.template_path:
            docs = self.generate_markdown_docs()
            with open(doc_output_dir / f"{self.name}.md", "w") as f:
                f.write(docs if docs.endswith("\n") else docs + "\n")
