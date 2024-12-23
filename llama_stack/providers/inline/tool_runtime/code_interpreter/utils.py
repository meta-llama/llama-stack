# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

DIR = os.path.dirname(os.path.realpath(__file__))
CODE_ENV_PREFIX_FILE = os.path.join(DIR, "code_env_prefix.py")
CODE_ENV_PREFIX = None


def get_code_env_prefix() -> str:
    global CODE_ENV_PREFIX

    if CODE_ENV_PREFIX is None:
        with open(CODE_ENV_PREFIX_FILE, "r") as f:
            CODE_ENV_PREFIX = f.read()

    return CODE_ENV_PREFIX
