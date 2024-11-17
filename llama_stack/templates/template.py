# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime

from io import StringIO

from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

import jinja2
import yaml
from pydantic import BaseModel, Field

from rich.console import Console
from rich.table import Table

from llama_stack.distribution.datatypes import (
    Api,
    BuildConfig,
    DistributionSpec,
    ModelInput,
    Provider,
    ShieldInput,
    StackRunConfig,
)
from llama_stack.distribution.distribution import get_provider_registry
from llama_stack.distribution.utils.dynamic import instantiate_class_type
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig


class RunConfigSettings(BaseModel):
    provider_overrides: Dict[str, List[Provider]] = Field(default_factory=dict)
    default_models: List[ModelInput]
    default_shields: Optional[List[ShieldInput]] = None

    def run_config(
        self,
        name: str,
        providers: Dict[str, List[str]],
        docker_image: Optional[str] = None,
    ) -> StackRunConfig:
        provider_registry = get_provider_registry()

        provider_configs = {}
        for api_str, provider_types in providers.items():
            if api_providers := self.provider_overrides.get(api_str):
                provider_configs[api_str] = api_providers
                continue

            provider_type = provider_types[0]
            provider_id = provider_type.split("::")[-1]

            api = Api(api_str)
            if provider_type not in provider_registry[api]:
                raise ValueError(
                    f"Unknown provider type: {provider_type} for API: {api_str}"
                )

            config_class = provider_registry[api][provider_type].config_class
            assert (
                config_class is not None
            ), f"No config class for provider type: {provider_type} for API: {api_str}"

            config_class = instantiate_class_type(config_class)
            if hasattr(config_class, "sample_run_config"):
                config = config_class.sample_run_config(
                    __distro_dir__=f"distributions/{name}"
                )
            else:
                config = {}

            provider_configs[api_str] = [
                Provider(
                    provider_id=provider_id,
                    provider_type=provider_type,
                    config=config,
                )
            ]

        # Get unique set of APIs from providers
        apis: Set[str] = set(providers.keys())

        return StackRunConfig(
            image_name=name,
            docker_image=docker_image,
            built_at=datetime.now(),
            apis=list(apis),
            providers=provider_configs,
            metadata_store=SqliteKVStoreConfig.sample_run_config(
                dir=f"distributions/{name}",
                db_name="registry.db",
            ),
            models=self.default_models,
            shields=self.default_shields or [],
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
    template_path: Path

    # Optional configuration
    docker_compose_env_vars: Optional[Dict[str, Tuple[str, str]]] = None
    docker_image: Optional[str] = None

    default_models: Optional[List[ModelInput]] = None

    def build_config(self) -> BuildConfig:
        return BuildConfig(
            name=self.name,
            distribution_spec=DistributionSpec(
                description=self.description,
                docker_image=self.docker_image,
                providers=self.providers,
            ),
            image_type="conda",  # default to conda, can be overridden
        )

    def generate_markdown_docs(self) -> str:
        """Generate markdown documentation using both Jinja2 templates and rich tables."""
        # First generate the providers table using rich
        output = StringIO()
        console = Console(file=output, force_terminal=False)

        table = Table(title="Provider Configuration", show_header=True)
        table.add_column("API", style="bold")
        table.add_column("Provider(s)")

        for api, providers in sorted(self.providers.items()):
            table.add_row(api, ", ".join(f"`{p}`" for p in providers))

        console.print(table)
        providers_table = output.getvalue()

        template = self.template_path.read_text()
        # Render template with rich-generated table
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        template = env.from_string(template)
        return template.render(
            name=self.name,
            description=self.description,
            providers=self.providers,
            providers_table=providers_table,
            docker_compose_env_vars=self.docker_compose_env_vars,
            default_models=self.default_models,
        )

    def save_distribution(self, yaml_output_dir: Path, doc_output_dir: Path) -> None:
        for output_dir in [yaml_output_dir, doc_output_dir]:
            output_dir.mkdir(parents=True, exist_ok=True)

        build_config = self.build_config()
        with open(yaml_output_dir / "build.yaml", "w") as f:
            yaml.safe_dump(build_config.model_dump(), f, sort_keys=False)

        for yaml_pth, settings in self.run_configs.items():
            run_config = settings.run_config(
                self.name, self.providers, self.docker_image
            )
            with open(yaml_output_dir / yaml_pth, "w") as f:
                yaml.safe_dump(run_config.model_dump(), f, sort_keys=False)

        docs = self.generate_markdown_docs()
        with open(doc_output_dir / f"{self.name}.md", "w") as f:
            f.write(docs)
