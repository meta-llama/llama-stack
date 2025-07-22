# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse


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
        dest="config",
        help="(DEPRECATED) Use positional argument [config] instead. Configuration file path",
    )

    group.add_argument(
        "--template",
        dest="config",
        help="(DEPRECATED) Use positional argument [config] instead. Template name",
    )
