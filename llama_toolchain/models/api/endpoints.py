# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol

from llama_models.schema_utils import webmethod  # noqa: F401

from pydantic import BaseModel  # noqa: F401


class Models(Protocol): ...
