# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import StrEnum
from pathlib import Path

from llama_stack.distribution.utils.config_dirs import DISTRIBS_BASE_DIR
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="config_resolution")


TEMPLATE_DIR = Path(__file__).parent.parent.parent.parent / "llama_stack" / "templates"


class Mode(StrEnum):
    RUN = "run"
    BUILD = "build"


def resolve_config_or_template(
    config_or_template: str,
    mode: Mode = Mode.RUN,
) -> Path:
    """
    Resolve a config/template argument to a concrete config file path.

    Args:
        config_or_template: User input (file path, template name, or built distribution)
        mode: Mode resolving for ("run", "build", "server")

    Returns:
        Path to the resolved config file

    Raises:
        ValueError: If resolution fails
    """

    # Strategy 1: Try as file path first
    config_path = Path(config_or_template)
    if config_path.exists() and config_path.is_file():
        logger.info(f"Using file path: {config_path}")
        return config_path.resolve()

    # Strategy 2: Try as template name (if no .yaml extension)
    if not config_or_template.endswith(".yaml"):
        template_config = _get_template_config_path(config_or_template, mode)
        if template_config.exists():
            logger.info(f"Using template: {template_config}")
            return template_config

    # Strategy 3: Try as built distribution name
    distrib_config = DISTRIBS_BASE_DIR / f"llamastack-{config_or_template}" / f"{config_or_template}-{mode}.yaml"
    if distrib_config.exists():
        logger.info(f"Using built distribution: {distrib_config}")
        return distrib_config

    distrib_config = DISTRIBS_BASE_DIR / f"{config_or_template}" / f"{config_or_template}-{mode}.yaml"
    if distrib_config.exists():
        logger.info(f"Using built distribution: {distrib_config}")
        return distrib_config

    # Strategy 4: Failed - provide helpful error
    raise ValueError(_format_resolution_error(config_or_template, mode))


def _get_template_config_path(template_name: str, mode: Mode) -> Path:
    """Get the config file path for a template."""
    return TEMPLATE_DIR / template_name / f"{mode}.yaml"


def _format_resolution_error(config_or_template: str, mode: Mode) -> str:
    """Format a helpful error message for resolution failures."""
    from llama_stack.distribution.utils.config_dirs import DISTRIBS_BASE_DIR

    template_path = _get_template_config_path(config_or_template, mode)
    distrib_path = DISTRIBS_BASE_DIR / f"llamastack-{config_or_template}" / f"{config_or_template}-{mode}.yaml"
    distrib_path2 = DISTRIBS_BASE_DIR / f"{config_or_template}" / f"{config_or_template}-{mode}.yaml"

    available_templates = _get_available_templates()
    templates_str = ", ".join(available_templates) if available_templates else "none found"

    return f"""Could not resolve config or template '{config_or_template}'.

Tried the following locations:
  1. As file path: {Path(config_or_template).resolve()}
  2. As template: {template_path}
  3. As built distribution: ({distrib_path}, {distrib_path2})

Available templates: {templates_str}

Did you mean one of these templates?
{_format_template_suggestions(available_templates, config_or_template)}
"""


def _get_available_templates() -> list[str]:
    """Get list of available template names."""
    if not TEMPLATE_DIR.exists() and not DISTRIBS_BASE_DIR.exists():
        return []

    return list(
        set(
            [d.name for d in TEMPLATE_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
            + [d.name for d in DISTRIBS_BASE_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
        )
    )


def _format_template_suggestions(templates: list[str], user_input: str) -> str:
    """Format template suggestions for error messages, showing closest matches first."""
    if not templates:
        return "  (no templates found)"

    import difflib

    # Get up to 3 closest matches with similarity threshold of 0.3 (lower = more permissive)
    close_matches = difflib.get_close_matches(user_input, templates, n=3, cutoff=0.3)
    display_templates = close_matches if close_matches else templates[:3]

    suggestions = [f"  - {t}" for t in display_templates]
    return "\n".join(suggestions)
