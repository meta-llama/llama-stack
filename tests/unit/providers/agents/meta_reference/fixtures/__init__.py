# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import yaml

from llama_stack.apis.inference.inference import (
    OpenAIChatCompletion,
)


def load_chat_completion_fixture(filename: str) -> OpenAIChatCompletion:
    """
    Load a YAML fixture file and convert it to an OpenAIChatCompletion object.

    Args:
        filename: Name of the YAML file (without path)

    Returns:
        OpenAIChatCompletion object
    """
    fixtures_dir = os.path.dirname(os.path.abspath(__file__))
    fixture_path = os.path.join(fixtures_dir, filename)

    with open(fixture_path) as f:
        data = yaml.safe_load(f)
    return OpenAIChatCompletion(**data)
