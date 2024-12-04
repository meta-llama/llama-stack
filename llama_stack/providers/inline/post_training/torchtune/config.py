# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from pydantic import BaseModel, Field


class TorchtunePostTrainingConfig(BaseModel):
    model: str = Field(
        default="Llama3.2-3B-Instruct",
        description="Model descriptor from `llama model list`",
    )
    torch_seed: Optional[int] = None
    # By default, the implementation will look at ~/.llama/checkpoints/<model> but you
    # can override by specifying the directory explicitly
    checkpoint_dir: Optional[str] = None
