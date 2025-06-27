#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import subprocess
import sys
from pathlib import Path
from typing import Any

from rich.progress import Progress, SpinnerColumn, TextColumn

from llama_stack.distribution.distribution import get_provider_registry

REPO_ROOT = Path(__file__).parent.parent


class ChangedPathTracker:
    """Track a list of paths we may have changed."""

    def __init__(self):
        self._changed_paths = []

    def add_paths(self, *paths):
        for path in paths:
            path = str(path)
            if path not in self._changed_paths:
                self._changed_paths.append(path)

    def changed_paths(self):
        return self._changed_paths


def get_config_class_info(config_class_path: str) -> dict[str, Any]:
    """Extract configuration information from a config class."""
    try:
        # Import the config class
        module_path, class_name = config_class_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        config_class = getattr(module, class_name)

        # Get the class docstring
        docstring = config_class.__doc__ or ""

        # Check if the model accepts extra fields
        accepts_extra_config = False
        try:
            # Try to get the model's JSON schema to see if it allows additional properties
            schema = config_class.model_json_schema()
            if schema.get("additionalProperties") is True:
                accepts_extra_config = True
        except Exception:
            # Fallback: check model_config attribute
            if hasattr(config_class, "model_config"):
                model_config = config_class.model_config
                # Check if it's a ConfigDict with extra="allow"
                if hasattr(model_config, "extra") and model_config.extra == "allow":
                    accepts_extra_config = True
                # Also check if it's a dict-like object with extra="allow"
                elif isinstance(model_config, dict) and model_config.get("extra") == "allow":
                    accepts_extra_config = True

        # Get field information from Pydantic model
        fields_info = {}
        if hasattr(config_class, "model_fields"):
            for field_name, field in config_class.model_fields.items():
                # Clean up the type annotation for display
                field_type = str(field.annotation) if field.annotation else "Any"
                field_type = field_type.replace("typing.", "").replace("Optional[", "").replace("]", "")
                field_type = field_type.replace("Annotated[", "").replace("FieldInfo(", "").replace(")", "")
                field_type = field_type.replace("llama_stack.apis.inference.inference.", "")
                field_type = field_type.replace("llama_stack.providers.", "")

                # Handle default values, including default_factory
                default_value = field.default
                if field.default_factory is not None:
                    try:
                        default_value = field.default_factory()
                    except Exception:
                        default_value = ""
                elif field.default is None:
                    default_value = ""

                field_info = {
                    "type": field_type,
                    "description": field.description or "",
                    "default": default_value,
                    "required": field.default is None and not field.is_required,
                }
                fields_info[field_name] = field_info

        # Add config field if the model accepts extra configuration
        if accepts_extra_config:
            # Try to extract comments above model_config line
            config_description = "Additional configuration options that will be forwarded to the underlying provider"
            try:
                import inspect

                # Get the source code of the class
                source = inspect.getsource(config_class)
                lines = source.split("\n")

                # Find the model_config line
                for i, line in enumerate(lines):
                    if "model_config" in line and "ConfigDict" in line and 'extra="allow"' in line:
                        # Look for comments above this line
                        comments = []
                        for j in range(i - 1, -1, -1):
                            stripped = lines[j].strip()
                            if stripped.startswith("#"):
                                comments.append(stripped[1:].strip())
                            elif stripped == "":
                                continue
                            else:
                                break

                        if comments:
                            config_description = " ".join(reversed(comments))
                        break
            except Exception:
                # If we can't extract comments, use the default description
                pass

            fields_info["config"] = {
                "type": "dict",
                "description": config_description,
                "default": "{}",
                "required": False,
            }

        return {
            "docstring": docstring,
            "fields": fields_info,
            "sample_config": getattr(config_class, "sample_run_config", None),
            "accepts_extra_config": accepts_extra_config,
        }
    except Exception as e:
        return {
            "error": f"Failed to load config class {config_class_path}: {str(e)}",
            "docstring": "",
            "fields": {},
            "sample_config": None,
            "accepts_extra_config": False,
        }


def generate_provider_docs(provider_spec: Any, api_name: str) -> str:
    """Generate markdown documentation for a provider."""
    provider_type = provider_spec.provider_type
    config_class = provider_spec.config_class

    # Get config information
    config_info = get_config_class_info(config_class)

    # Generate markdown
    md_lines = []
    md_lines.append(f"# {provider_type}")
    md_lines.append("")

    # Use the description from the provider spec if available, otherwise fall back to config docstring
    description = ""
    if hasattr(provider_spec, "description") and provider_spec.description:
        description = provider_spec.description
    elif (
        hasattr(provider_spec, "adapter")
        and hasattr(provider_spec.adapter, "description")
        and provider_spec.adapter.description
    ):
        description = provider_spec.adapter.description
    elif config_info.get("docstring"):
        description = config_info["docstring"]

    if description:
        md_lines.append("## Description")
        md_lines.append("")
        md_lines.append(description)
        md_lines.append("")

    if config_info.get("fields"):
        md_lines.append("## Configuration")
        md_lines.append("")
        md_lines.append("| Field | Type | Required | Default | Description |")
        md_lines.append("|-------|------|----------|---------|-------------|")

        for field_name, field_info in config_info["fields"].items():
            field_type = field_info["type"].replace("|", "\\|")
            required = "Yes" if field_info["required"] else "No"
            default = str(field_info["default"]) if field_info["default"] is not None else ""
            description = field_info["description"] or ""

            md_lines.append(f"| `{field_name}` | `{field_type}` | {required} | {default} | {description} |")

        md_lines.append("")

        # Add note about accepting extra configuration if applicable
        if config_info.get("accepts_extra_config"):
            md_lines.append(
                "> **Note**: This configuration class accepts additional fields beyond those listed above. You can pass any additional configuration options that will be forwarded to the underlying provider."
            )
            md_lines.append("")

    if config_info.get("sample_config"):
        md_lines.append("## Sample Configuration")
        md_lines.append("")
        md_lines.append("```yaml")
        try:
            sample_config_func = config_info["sample_config"]
            import inspect

            import yaml

            if sample_config_func is not None:
                sig = inspect.signature(sample_config_func)
                if "__distro_dir__" in sig.parameters:
                    sample_config = sample_config_func(__distro_dir__="~/.llama/dummy")
                else:
                    sample_config = sample_config_func()

                # Convert Pydantic objects to dictionaries for proper YAML serialization
                def convert_pydantic_to_dict(obj):
                    if hasattr(obj, "model_dump"):
                        return obj.model_dump()
                    elif hasattr(obj, "dict"):
                        return obj.dict()
                    elif isinstance(obj, dict):
                        return {k: convert_pydantic_to_dict(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_pydantic_to_dict(item) for item in obj]
                    else:
                        return obj

                sample_config_dict = convert_pydantic_to_dict(sample_config)
                md_lines.append(yaml.dump(sample_config_dict, default_flow_style=False, sort_keys=False))
            else:
                md_lines.append("# No sample configuration available.")
        except Exception as e:
            md_lines.append(f"# Error generating sample config: {str(e)}")
        md_lines.append("```")
        md_lines.append("")

    if hasattr(provider_spec, "deprecation_warning") and provider_spec.deprecation_warning:
        md_lines.append("## Deprecation Notice")
        md_lines.append("")
        md_lines.append(f"⚠️ **Warning**: {provider_spec.deprecation_warning}")
        md_lines.append("")

    if hasattr(provider_spec, "deprecation_error") and provider_spec.deprecation_error:
        md_lines.append("## Deprecation Error")
        md_lines.append("")
        md_lines.append(f"❌ **Error**: {provider_spec.deprecation_error}")

    return "\n".join(md_lines) + "\n"


def process_provider_registry(progress, change_tracker: ChangedPathTracker) -> None:
    """Process the complete provider registry."""
    progress.print("Processing provider registry")

    try:
        # Get the complete provider registry
        provider_registry = get_provider_registry()

        # Generate documentation for each API
        for api, providers in provider_registry.items():
            api_name = api.value

            # Create API-specific documentation directory
            doc_output_dir = REPO_ROOT / "docs" / "source" / "providers" / api_name
            doc_output_dir.mkdir(parents=True, exist_ok=True)
            change_tracker.add_paths(doc_output_dir)

            # Generate index file for the API
            index_content = []
            index_content.append(f"# {api_name.title()} Providers")
            index_content.append("")
            index_content.append(
                f"This section contains documentation for all available providers for the **{api_name}** API."
            )
            index_content.append("")

            for provider_type, provider in sorted(providers.items()):
                provider_doc_file = doc_output_dir / f"{provider_type.replace('::', '_').replace(':', '_')}.md"

                # Generate provider documentation
                provider_docs = generate_provider_docs(provider, api_name)

                # Write provider documentation file
                provider_doc_file.write_text(provider_docs)
                change_tracker.add_paths(provider_doc_file)

                # Add to index
                index_content.append(f"- [{provider_type}]({provider_doc_file.name})")

            # Write index file
            index_file = doc_output_dir / "index.md"
            index_file.write_text("\n".join(index_content))
            change_tracker.add_paths(index_file)

    except Exception as e:
        progress.print(f"[red]Error processing provider registry: {str(e)}")
        raise e


def check_for_changes(change_tracker: ChangedPathTracker) -> bool:
    """Check if there are any uncommitted changes, including new files."""
    has_changes = False
    for path in change_tracker.changed_paths():
        # Check for modified files
        result = subprocess.run(
            ["git", "diff", "--exit-code", path],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"Change detected in '{path}'.", file=sys.stderr)
            has_changes = True
        # Check for untracked (new) files
        status_result = subprocess.run(
            ["git", "status", "--porcelain", path],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        for line in status_result.stdout.splitlines():
            if line.startswith("??"):
                print(f"New file detected: '{path}'.", file=sys.stderr)
                has_changes = True
    return has_changes


def main():
    change_tracker = ChangedPathTracker()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        task = progress.add_task("Processing provider registry...", total=1)

        process_provider_registry(progress, change_tracker)
        progress.update(task, advance=1)

    if check_for_changes(change_tracker):
        print(
            "Provider documentation changes detected. Please commit the changes.",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
