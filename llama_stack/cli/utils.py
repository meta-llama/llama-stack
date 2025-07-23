# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="cli")


def add_config_template_args(parser: argparse.ArgumentParser):
    """Add unified config/template arguments with backward compatibility."""
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "config",
        nargs="?",
        help="Configuration file path or template name",
    )

    # Backward compatibility arguments (deprecated)
    group.add_argument(
        "--config",
        dest="config_deprecated",
        help="(DEPRECATED) Use positional argument [config] instead. Configuration file path",
    )

    group.add_argument(
        "--template",
        dest="template_deprecated",
        help="(DEPRECATED) Use positional argument [config] instead. Template name",
    )


def get_config_from_args(args: argparse.Namespace) -> str | None:
    """Extract config value from parsed arguments, handling both new and deprecated forms."""
    if args.config is not None:
        return str(args.config)
    elif hasattr(args, "config_deprecated") and args.config_deprecated is not None:
        logger.warning("Using deprecated --config argument. Use positional argument [config] instead.")
        return str(args.config_deprecated)
    elif hasattr(args, "template_deprecated") and args.template_deprecated is not None:
        logger.warning("Using deprecated --template argument. Use positional argument [config] instead.")
        return str(args.template_deprecated)
    return None
