#!/usr/bin/env python
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
from typing import Iterable

from rich.progress import Progress, SpinnerColumn, TextColumn

from llama_stack.distribution.build import (
    SERVER_DEPENDENCIES,
    get_provider_dependencies,
)

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


def find_template_dirs(templates_dir: Path) -> Iterable[Path]:
    """Find immediate subdirectories in the templates folder."""
    if not templates_dir.exists():
        raise FileNotFoundError(f"Templates directory not found: {templates_dir}")

    return sorted(d for d in templates_dir.iterdir() if d.is_dir() and d.name != "__pycache__")


def process_template(template_dir: Path, progress, change_tracker: ChangedPathTracker) -> None:
    """Process a single template directory."""
    progress.print(f"Processing {template_dir.name}")

    try:
        # Import the module directly
        module_name = f"llama_stack.templates.{template_dir.name}"
        module = importlib.import_module(module_name)

        # Get and save the distribution template
        if template_func := getattr(module, "get_distribution_template", None):
            template = template_func()

            yaml_output_dir = REPO_ROOT / "llama_stack" / "templates" / template.name
            doc_output_dir = REPO_ROOT / "docs/source/distributions" / f"{template.distro_type}_distro"
            change_tracker.add_paths(yaml_output_dir, doc_output_dir)
            template.save_distribution(
                yaml_output_dir=yaml_output_dir,
                doc_output_dir=doc_output_dir,
            )
        else:
            progress.print(f"[yellow]Warning: {template_dir.name} has no get_distribution_template function")

    except Exception as e:
        progress.print(f"[red]Error processing {template_dir.name}: {str(e)}")
        raise e


def check_for_changes(change_tracker: ChangedPathTracker) -> bool:
    """Check if there are any uncommitted changes."""
    has_changes = False
    for path in change_tracker.changed_paths():
        result = subprocess.run(
            ["git", "diff", "--exit-code", path],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"Change detected in '{path}'.", file=sys.stderr)
            has_changes = True
    return has_changes


def collect_template_dependencies(template_dir: Path) -> tuple[str | None, list[str]]:
    try:
        module_name = f"llama_stack.templates.{template_dir.name}"
        module = importlib.import_module(module_name)

        if template_func := getattr(module, "get_distribution_template", None):
            template = template_func()
            normal_deps, special_deps = get_provider_dependencies(template.providers)
            # Combine all dependencies in order: normal deps, special deps, server deps
            all_deps = sorted(set(normal_deps + SERVER_DEPENDENCIES)) + sorted(set(special_deps))

            return template.name, all_deps
    except Exception:
        return None, []
    return None, []


def generate_dependencies_file(change_tracker: ChangedPathTracker):
    templates_dir = REPO_ROOT / "llama_stack" / "templates"
    distribution_deps = {}

    for template_dir in find_template_dirs(templates_dir):
        name, deps = collect_template_dependencies(template_dir)
        if name:
            distribution_deps[name] = deps

    deps_file = REPO_ROOT / "llama_stack" / "templates" / "dependencies.json"
    change_tracker.add_paths(deps_file)
    with open(deps_file, "w") as f:
        f.write(json.dumps(distribution_deps, indent=2) + "\n")


def main():
    templates_dir = REPO_ROOT / "llama_stack" / "templates"
    change_tracker = ChangedPathTracker()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        template_dirs = list(find_template_dirs(templates_dir))
        task = progress.add_task("Processing distribution templates...", total=len(template_dirs))

        # Create a partial function with the progress bar
        process_func = partial(process_template, progress=progress, change_tracker=change_tracker)

        # Process templates in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tasks and wait for completion
            list(executor.map(process_func, template_dirs))
            progress.update(task, advance=len(template_dirs))

    generate_dependencies_file(change_tracker)

    if check_for_changes(change_tracker):
        print(
            "Distribution template changes detected. Please commit the changes.",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
