# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Literal, Optional

from pydantic import BaseModel


class HFilabPostTrainingConfig(BaseModel):
    torch_seed: Optional[int] = None
    checkpoint_format: Optional[Literal["meta", "huggingface"]] = "meta"
