# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from io import StringIO

from pathlib import Path
from typing import Dict, List, Optional, Set

import jinja2
import yaml
from pydantic import BaseModel

from rich.console import Console
from rich.table import Table

from llama_stack.distribution.datatypes import (
    BuildConfig,
    DistributionSpec,
    KVStoreConfig,
    ModelInput,
    Provider,
    ShieldInput,
    StackRunConfig,
)


class DistributionTemplate(BaseModel):
    """
    Represents a Llama Stack distribution instance that can generate configuration
    and documentation files.
    """

    name: str
    description: str
    providers: Dict[str, List[str]]
    default_models: List[ModelInput]
    default_shields: Optional[List[ShieldInput]] = None

    # Optional configuration
    metadata_store: Optional[KVStoreConfig] = None
    env_vars: Optional[Dict[str, str]] = None
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

    def run_config(self, provider_configs: Dict[str, List[Provider]]) -> StackRunConfig:
        from datetime import datetime

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

{% for var, description in env_vars.items() %}
- `{{ var }}`: {{ description }}
{% endfor %}
{%- endif %}

## Example Usage

### Using Docker Compose

```bash
$ cd distributions/{{ name }}
$ docker compose up
```

### Manual Configuration

You can also configure the distribution manually by creating a `run.yaml` file:

```yaml
version: '2'
image_name: {{ name }}
apis:
{% for api in providers.keys() %}
  - {{ api }}
{% endfor %}

providers:
{% for api, provider_list in providers.items() %}
  {{ api }}:
  {% for provider in provider_list %}
    - provider_id: {{ provider.lower() }}-0
      provider_type: {{ provider }}
      config: {}
  {% endfor %}
{% endfor %}
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
            env_vars=self.env_vars,
            default_models=self.default_models,
            default_shields=self.default_shields,
        )

    def save_distribution(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save build.yaml
        build_config = self.build_config()
        with open(output_dir / "build.yaml", "w") as f:
            yaml.safe_dump(build_config.model_dump(), f, sort_keys=False)

        # Save run.yaml template
        # Create a minimal provider config for the template
        provider_configs = {
            api: [
                Provider(
                    provider_id=f"{provider.lower()}-0",
                    provider_type=provider,
                    config={},
                )
                for provider in providers
            ]
            for api, providers in self.providers.items()
        }
        run_config = self.run_config(provider_configs)
        with open(output_dir / "run.yaml", "w") as f:
            yaml.safe_dump(run_config.model_dump(), f, sort_keys=False)

        # Save documentation
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
            default_models=[
                ModelInput(
                    model_id="${env.LLAMA_INFERENCE_MODEL:Llama3.1-8B-Instruct}"
                ),
                ModelInput(model_id="${env.LLAMA_SAFETY_MODEL:Llama-Guard-3-1B}"),
            ],
            default_shields=[
                ShieldInput(shield_id="${env.LLAMA_SAFETY_MODEL:Llama-Guard-3-1B}")
            ],
            env_vars={
                "LLAMA_INFERENCE_VLLM_URL": "URL of the vLLM inference server",
                "LLAMA_SAFETY_VLLM_URL": "URL of the vLLM safety server",
                "MAX_TOKENS": "Maximum number of tokens for generation",
                "LLAMA_INFERENCE_MODEL": "Name of the inference model to use",
                "LLAMA_SAFETY_MODEL": "Name of the safety model to use",
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
