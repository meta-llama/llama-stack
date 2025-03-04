# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from pydantic import BaseModel


class InclineSimpleChunkingConfig(BaseModel):
    chunk_size_in_tokens: int = 512
    chunk_overlap_ratio: int = 4
