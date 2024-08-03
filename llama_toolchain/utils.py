# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os
from enum import Enum
from pathlib import Path


LLAMA_STACK_CONFIG_DIR = os.path.expanduser("~/.llama/")

DISTRIBS_BASE_DIR = Path(LLAMA_STACK_CONFIG_DIR) / "distributions"


def get_root_directory():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while os.path.isfile(os.path.join(current_dir, "__init__.py")):
        current_dir = os.path.dirname(current_dir)

    return current_dir


def get_default_config_dir():
    return os.path.join(LLAMA_STACK_CONFIG_DIR, "configs")


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)
