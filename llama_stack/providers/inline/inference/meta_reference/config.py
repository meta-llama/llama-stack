# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, Optional

from pydantic import BaseModel, field_validator

from llama_stack.apis.inference import QuantizationConfig
from llama_stack.providers.utils.inference import supported_inference_models


class MetaReferenceInferenceConfig(BaseModel):
    # this is a placeholder to indicate inference model id
    # the actual inference model id is dtermined by the moddel id in the request
    # Note: you need to register the model before using it for inference
    # models in the resouce list in the run.yaml config will be registered automatically
    model: Optional[str] = None
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
        descriptors = [m.descriptor() for m in permitted_models]
        repos = [m.huggingface_repo for m in permitted_models]
        if model not in (descriptors + repos):
            model_list = "\n\t".join(repos)
            raise ValueError(f"Unknown model: `{model}`. Choose from [\n\t{model_list}\n]")
        return model

    @classmethod
    def sample_run_config(
        cls,
        model: str = "Llama3.2-3B-Instruct",
        checkpoint_dir: str = "${env.CHECKPOINT_DIR:null}",
        **kwargs,
    ) -> Dict[str, Any]:
        return {
            "model": model,
            "max_seq_len": 4096,
            "checkpoint_dir": checkpoint_dir,
        }


class MetaReferenceQuantizedInferenceConfig(MetaReferenceInferenceConfig):
    quantization: QuantizationConfig

    @classmethod
    def sample_run_config(
        cls,
        model: str = "Llama3.2-3B-Instruct",
        checkpoint_dir: str = "${env.CHECKPOINT_DIR:null}",
        **kwargs,
    ) -> Dict[str, Any]:
        config = super().sample_run_config(model, checkpoint_dir, **kwargs)
        config["quantization"] = {
            "type": "fp8",
        }
        return config
