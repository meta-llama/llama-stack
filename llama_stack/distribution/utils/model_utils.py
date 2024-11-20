# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

from .config_dirs import DEFAULT_CHECKPOINT_DIR


def model_local_dir(descriptor: str) -> str:
    path = os.path.join(DEFAULT_CHECKPOINT_DIR, descriptor)
    return path.replace(":", "-")
