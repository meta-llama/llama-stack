# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from pydantic import BaseModel


class LlamaGuardShieldConfig(BaseModel):
    model_dir: str
    excluded_categories: List[str]
    disable_input_check: bool = False
    disable_output_check: bool = False


class PromptGuardShieldConfig(BaseModel):
    model_dir: str


class SafetyConfig(BaseModel):
    llama_guard_shield: Optional[LlamaGuardShieldConfig] = None
    prompt_guard_shield: Optional[PromptGuardShieldConfig] = None
