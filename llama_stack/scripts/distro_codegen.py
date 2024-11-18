# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import concurrent.futures
import importlib
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Iterator

from rich.progress import Progress, SpinnerColumn, TextColumn


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
                / "docs/source/getting_started/distributions"
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

    if check_for_changes():
        print(
            "Distribution template changes detected. Please commit the changes.",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
