# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from pydantic import BaseModel


class LlamaGuardConfig(BaseModel):
    excluded_categories: List[str] = []
