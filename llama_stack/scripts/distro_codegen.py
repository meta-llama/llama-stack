# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import concurrent.futures
import importlib
import json
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Iterator

from rich.progress import Progress, SpinnerColumn, TextColumn

from llama_stack.distribution.build import (
    get_provider_dependencies,
    SERVER_DEPENDENCIES,
)


REPO_ROOT = Path(__file__).parent.parent.parent


def find_template_dirs(templates_dir: Path) -> Iterator[Path]:
    """Find immediate subdirectories in the templates folder."""
    if not templates_dir.exists():
        raise FileNotFoundError(f"Templates directory not found: {templates_dir}")

    return (
        d for d in templates_dir.iterdir() if d.is_dir() and d.name != "__pycache__"
    )


def process_template(template_dir: Path, progress) -> None:
    """Process a single template directory."""
    progress.print(f"Processing {template_dir.name}")

    try:
        # Import the module directly
        module_name = f"llama_stack.templates.{template_dir.name}"
        module = importlib.import_module(module_name)

        # Get and save the distribution template
        if template_func := getattr(module, "get_distribution_template", None):
            template = template_func()

            template.save_distribution(
                yaml_output_dir=REPO_ROOT / "llama_stack" / "templates" / template.name,
                doc_output_dir=REPO_ROOT
                / "docs/source/distributions"
                / f"{template.distro_type}_distro",
            )
        else:
            progress.print(
                f"[yellow]Warning: {template_dir.name} has no get_distribution_template function"
            )

    except Exception as e:
        progress.print(f"[red]Error processing {template_dir.name}: {str(e)}")
        raise e


def check_for_changes() -> bool:
    """Check if there are any uncommitted changes."""
    result = subprocess.run(
        ["git", "diff", "--exit-code"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return result.returncode != 0


def collect_template_dependencies(template_dir: Path) -> tuple[str, list[str]]:
    try:
        module_name = f"llama_stack.templates.{template_dir.name}"
        module = importlib.import_module(module_name)

        if template_func := getattr(module, "get_distribution_template", None):
            template = template_func()
            normal_deps, special_deps = get_provider_dependencies(template.providers)
            # Combine all dependencies in order: normal deps, special deps, server deps
            all_deps = sorted(list(set(normal_deps + SERVER_DEPENDENCIES))) + sorted(
                list(set(special_deps))
            )

            return template.name, all_deps
    except Exception:
        return None, []
    return None, []


def generate_dependencies_file():
    templates_dir = REPO_ROOT / "llama_stack" / "templates"
    distribution_deps = {}

    for template_dir in find_template_dirs(templates_dir):
        name, deps = collect_template_dependencies(template_dir)
        if name:
            distribution_deps[name] = deps

    deps_file = REPO_ROOT / "distributions" / "dependencies.json"
    with open(deps_file, "w") as f:
        f.write(json.dumps(distribution_deps, indent=2) + "\n")


def main():
    templates_dir = REPO_ROOT / "llama_stack" / "templates"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        template_dirs = list(find_template_dirs(templates_dir))
        task = progress.add_task(
            "Processing distribution templates...", total=len(template_dirs)
        )

        # Create a partial function with the progress bar
        process_func = partial(process_template, progress=progress)

        # Process templates in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tasks and wait for completion
            list(executor.map(process_func, template_dirs))
            progress.update(task, advance=len(template_dirs))

    generate_dependencies_file()

    if check_for_changes():
        print(
            "Distribution template changes detected. Please commit the changes.",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
