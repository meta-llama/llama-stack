# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from llama_models.datatypes import *  # noqa: F403
from llama_models.sku_list import resolve_model

from llama_stack.apis.inference import *  # noqa: F401, F403
from pydantic import BaseModel, Field, field_validator

from llama_stack.providers.utils.inference import supported_inference_models


class MetaReferenceInferenceConfig(BaseModel):
    model: str = Field(
        default="Llama3.2-3B-Instruct",
        description="Model descriptor from `llama model list`",
    )
    torch_seed: Optional[int] = None
    max_seq_len: int = 4096
    max_batch_size: int = 1

    # when this is False, we assume that the distributed process group is setup by someone
    # outside of this code (e.g., when run inside `torchrun`). that is useful for clients
    # (including our testing code) who might be using llama-stack as a library.
    create_distributed_process_group: bool = True

    # By default, the implementation will look at ~/.llama/checkpoints/<model> but you
    # can override by specifying the directory explicitly
    checkpoint_dir: Optional[str] = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, model: str) -> str:
        permitted_models = supported_inference_models()
        if model not in permitted_models:
            model_list = "\n\t".join(permitted_models)
            raise ValueError(
                f"Unknown model: `{model}`. Choose from [\n\t{model_list}\n]"
            )
        return model

    @property
    def model_parallel_size(self) -> int:
        resolved = resolve_model(self.model)
        return resolved.pth_file_count


class MetaReferenceQuantizedInferenceConfig(MetaReferenceInferenceConfig):
    quantization: QuantizationConfig
