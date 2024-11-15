# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime

from io import StringIO

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import jinja2
import yaml
from pydantic import BaseModel, Field

from rich.console import Console
from rich.table import Table

from llama_stack.distribution.datatypes import (
    Api,
    BuildConfig,
    DistributionSpec,
    KVStoreConfig,
    ModelInput,
    Provider,
    ShieldInput,
    StackRunConfig,
)
from llama_stack.distribution.distribution import get_provider_registry
from llama_stack.distribution.utils.dynamic import instantiate_class_type
from llama_stack.providers.remote.inference.vllm.config import (
    VLLMInferenceAdapterConfig,
)
from llama_stack.providers.utils.docker.service_config import DockerComposeServiceConfig


class DistributionTemplate(BaseModel):
    """
    Represents a Llama Stack distribution instance that can generate configuration
    and documentation files.
    """

    name: str
    description: str
    providers: Dict[str, List[str]]
    run_config_overrides: Dict[str, List[Provider]] = Field(default_factory=dict)
    compose_config_overrides: Dict[str, Dict[str, DockerComposeServiceConfig]] = Field(
        default_factory=dict
    )

    default_models: List[ModelInput]
    default_shields: Optional[List[ShieldInput]] = None

    # Optional configuration
    metadata_store: Optional[KVStoreConfig] = None
    docker_compose_env_vars: Optional[Dict[str, Tuple[str, str]]] = None
    docker_image: Optional[str] = None

    @property
    def distribution_spec(self) -> DistributionSpec:
        return DistributionSpec(
            description=self.description,
            docker_image=self.docker_image,
            providers=self.providers,
        )

    def build_config(self) -> BuildConfig:
        return BuildConfig(
            name=self.name,
            distribution_spec=self.distribution_spec,
            image_type="conda",  # default to conda, can be overridden
        )

    def run_config(self) -> StackRunConfig:
        provider_registry = get_provider_registry()

        provider_configs = {}
        for api_str, provider_types in self.providers.items():
            if providers := self.run_config_overrides.get(api_str):
                provider_configs[api_str] = providers
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
                config = config_class.sample_run_config()
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
        apis: Set[str] = set(self.providers.keys())

        return StackRunConfig(
            image_name=self.name,
            docker_image=self.docker_image,
            built_at=datetime.now(),
            apis=list(apis),
            providers=provider_configs,
            metadata_store=self.metadata_store,
            models=self.default_models,
            shields=self.default_shields or [],
        )

    def docker_compose_config(self) -> Dict[str, Any]:
        services = {}
        provider_registry = get_provider_registry()

        # Add provider services based on their sample_compose_config
        for api_str, api_providers in self.providers.items():
            if overrides := self.compose_config_overrides.get(api_str):
                services |= overrides
                continue

            # only look at the first provider to get the compose config for now
            # we may want to use `docker compose profiles` in the future
            provider_type = api_providers[0]
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
            if not hasattr(config_class, "sample_docker_compose_config"):
                continue

            compose_config = config_class.sample_docker_compose_config()
            services[provider_id] = compose_config

        port = "${LLAMASTACK_PORT:-5001}"
        # Add main llamastack service
        llamastack_config = DockerComposeServiceConfig(
            image=f"llamastack/distribution-{self.name}:latest",
            depends_on=list(services.keys()),
            volumes=[
                "~/.llama:/root/.llama",
                f"~/local/llama-stack/distributions/{self.name}/run.yaml:/root/llamastack-run-{self.name}.yaml",
            ],
            ports=[f"{port}:{port}"],
            environment={
                k: v[0] for k, v in (self.docker_compose_env_vars or {}).items()
            },
            entrypoint=(
                f'bash -c "sleep 60; python -m llama_stack.distribution.server.server --yaml_config /root/llamastack-run-{self.name}.yaml --port {port}"'
            ),
            deploy={
                "restart_policy": {
                    "condition": "on-failure",
                    "delay": "3s",
                    "max_attempts": 5,
                    "window": "60s",
                }
            },
        )

        services["llamastack"] = llamastack_config
        return {
            "services": {k: v.model_dump() for k, v in services.items()},
            "volumes": {service_name: None for service_name in services.keys()},
        }

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

        # Main documentation template
        template = """# {{ name }} Distribution

{{ description }}

## Provider Configuration

The `llamastack/distribution-{{ name }}` distribution consists of the following provider configurations:

{{ providers_table }}

{%- if env_vars %}
## Environment Variables

The following environment variables can be configured:

{% for var, (value, description) in docker_compose_env_vars.items() %}
- `{{ var }}`: {{ description }}
{% endfor %}
{%- endif %}

## Example Usage

### Using Docker Compose

```bash
$ cd distributions/{{ name }}
$ docker compose up
```

## Models

The following models are configured by default:
{% for model in default_models %}
- `{{ model.model_id }}`
{% endfor %}

{%- if default_shields %}

## Safety Shields

The following safety shields are configured:
{% for shield in default_shields %}
- `{{ shield.shield_id }}`
{%- endfor %}
{%- endif %}
"""
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
            default_shields=self.default_shields,
        )

    def save_distribution(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        build_config = self.build_config()
        with open(output_dir / "build.yaml", "w") as f:
            yaml.safe_dump(build_config.model_dump(), f, sort_keys=False)

        run_config = self.run_config()
        serialized = run_config.model_dump()
        with open(output_dir / "run.yaml", "w") as f:
            yaml.safe_dump(serialized, f, sort_keys=False)

        # serialized_str = yaml.dump(serialized, sort_keys=False)
        # env_vars = set()
        # for match in re.finditer(r"\${env\.([A-Za-z0-9_-]+)}", serialized_str):
        #     env_vars.add(match.group(1))

        docker_compose = self.docker_compose_config()
        with open(output_dir / "compose.yaml", "w") as f:
            yaml.safe_dump(docker_compose, f, sort_keys=False, default_flow_style=False)

        docs = self.generate_markdown_docs()
        with open(output_dir / f"{self.name}.md", "w") as f:
            f.write(docs)

    @classmethod
    def vllm_distribution(cls) -> "DistributionTemplate":
        return cls(
            name="remote-vllm",
            description="Use (an external) vLLM server for running LLM inference",
            providers={
                "inference": ["remote::vllm"],
                "memory": ["inline::faiss", "remote::chromadb", "remote::pgvector"],
                "safety": ["inline::llama-guard"],
                "agents": ["inline::meta-reference"],
                "telemetry": ["inline::meta-reference"],
            },
            run_config_overrides={
                "inference": [
                    Provider(
                        provider_id="vllm-0",
                        provider_type="remote::vllm",
                        config=VLLMInferenceAdapterConfig.sample_run_config(
                            url="${env.VLLM_URL:http://host.docker.internal:5100/v1}",
                        ),
                    ),
                    Provider(
                        provider_id="vllm-1",
                        provider_type="remote::vllm",
                        config=VLLMInferenceAdapterConfig.sample_run_config(
                            url="${env.SAFETY_VLLM_URL:http://host.docker.internal:5101/v1}",
                        ),
                    ),
                ]
            },
            compose_config_overrides={
                "inference": {
                    "vllm-0": VLLMInferenceAdapterConfig.sample_docker_compose_config(
                        port=5100,
                        cuda_visible_devices="0",
                        model="${env.INFERENCE_MODEL:Llama3.2-3B-Instruct}",
                    ),
                    "vllm-1": VLLMInferenceAdapterConfig.sample_docker_compose_config(
                        port=5100,
                        cuda_visible_devices="1",
                        model="${env.SAFETY_MODEL:Llama-Guard-3-1B}",
                    ),
                }
            },
            default_models=[
                ModelInput(
                    model_id="${env.INFERENCE_MODEL:Llama3.2-3B-Instruct}",
                    provider_id="vllm-0",
                ),
                ModelInput(
                    model_id="${env.SAFETY_MODEL:Llama-Guard-3-1B}",
                    provider_id="vllm-1",
                ),
            ],
            default_shields=[
                ShieldInput(shield_id="${env.SAFETY_MODEL:Llama-Guard-3-1B}")
            ],
            docker_compose_env_vars={
                # these defaults are for the Docker Compose configuration
                "VLLM_URL": (
                    "http://host.docker.internal:${VLLM_PORT:-5100}/v1",
                    "URL of the vLLM server with the main inference model",
                ),
                "SAFETY_VLLM_URL": (
                    "http://host.docker.internal:${SAFETY_VLLM_PORT:-5101}/v1",
                    "URL of the vLLM server with the safety model",
                ),
                "MAX_TOKENS": (
                    "${MAX_TOKENS:-4096}",
                    "Maximum number of tokens for generation",
                ),
                "INFERENCE_MODEL": (
                    "${INFERENCE_MODEL:-Llama3.2-3B-Instruct}",
                    "Name of the inference model to use",
                ),
                "SAFETY_MODEL": (
                    "${SAFETY_MODEL:-Llama-Guard-3-1B}",
                    "Name of the safety (Llama-Guard) model to use",
                ),
                "LLAMASTACK_PORT": (
                    "${LLAMASTACK_PORT:-5001}",
                    "Port for the Llama Stack distribution server",
                ),
            },
        )


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate a distribution template")
    parser.add_argument(
        "--type",
        choices=["vllm"],
        default="vllm",
        help="Type of distribution template to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the distribution files",
    )

    args = parser.parse_args()

    if args.type == "vllm":
        template = DistributionTemplate.vllm_distribution()
    else:
        print(f"Unknown template type: {args.type}", file=sys.stderr)
        sys.exit(1)

    template.save_distribution(args.output_dir)
