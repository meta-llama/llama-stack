# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from pydantic import BaseModel
from strong_typing.schema import json_schema_type

from llama_toolchain.inference.api import QuantizationConfig


@json_schema_type
class MetaReferenceImplConfig(BaseModel):
    model: str
    quantization: Optional[QuantizationConfig] = None
    torch_seed: Optional[int] = None
    max_seq_len: int
    max_batch_size: int = 1
