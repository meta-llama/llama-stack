# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import torch
from torchtune import training
from torchtune.models import convert_weights
from torchtune.training.checkpointing._utils import ModelType, safe_torch_load
from torchtune.utils._logging import get_logger

logger = get_logger("DEBUG")


class TorchtuneCheckpointer:
    def __init__(
        self,
        model_id: str,
        training_algorithm: str,
        checkpoint_dir: str,
        checkpoint_files: List[str],
        output_dir: str,
        model_type: str,
    ) -> None:
        # Fail fast if ``checkpoint_files`` is invalid
        # TODO: support loading more than one file
        if len(checkpoint_files) != 1:
            raise ValueError(
                "Currently we only support reading from a single torchtune checkpoint file. "
                f"Got {len(checkpoint_files)} files instead."
            )
        self._checkpoint_file = checkpoint_files[0]
        self._model_id = model_id
        self._training_algorithm = training_algorithm
        self._checkpoint_dir = Path(checkpoint_dir)
        self._model_type = ModelType[model_type]
        self._output_dir = output_dir
        # get ckpt paths
        self._checkpoint_path = Path.joinpath(
            self._checkpoint_dir, self._checkpoint_file
        )

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load Meta checkpoint from file. Currently only loading from a single file is supported.
        """
        state_dict: Dict[str:Any] = {}
        model_state_dict = safe_torch_load(self._checkpoint_path)
        if self._model_type == ModelType.LLAMA3_VISION:
            from torchtune.models.llama3_2_vision._convert_weights import (
                llama3_vision_meta_to_tune,
            )

            state_dict[training.MODEL_KEY] = llama3_vision_meta_to_tune(
                model_state_dict
            )
        else:
            state_dict[training.MODEL_KEY] = convert_weights.meta_to_tune(
                model_state_dict
            )

        # llama3_2 has tied weights, so we need to remove the output.weight key
        if self._model_type == ModelType.LLAMA3_2:
            logger.info(
                "Identified model_type = Llama3_2. Ignoring output.weight in"
                " checkpoint in favor of the tok_embedding.weight"
                " tied weights."
            )
            state_dict[training.MODEL_KEY].pop("output.weight")

        return state_dict

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        adapter_only: bool = False,
    ) -> str:
        model_file_path = (
            Path(self._output_dir)
            / f"{self._model_id}-{self._training_algorithm}-{epoch}"
        )

        model_file_path.mkdir(parents=True, exist_ok=True)

        # copy the related files for inference
        source_path = Path.joinpath(self._checkpoint_dir, "params.json")
        if source_path.exists():
            shutil.copy(
                source_path,
                Path.joinpath(model_file_path, "params.json"),
            )
        source_path = Path.joinpath(self._checkpoint_dir, "tokenizer.model")
        if source_path.exists():
            shutil.copy(
                source_path,
                Path.joinpath(model_file_path, "tokenizer.model"),
            )
        source_path = Path.joinpath(self._checkpoint_dir, "orig_params.json")
        if source_path.exists():
            shutil.copy(
                source_path,
                Path.joinpath(model_file_path, "orig_params.json"),
            )

        if not adapter_only:
            model_state_dict = state_dict[training.MODEL_KEY]
            if self._model_type == ModelType.LLAMA3_VISION:
                from torchtune.models.llama3_2_vision._convert_weights import (
                    llama3_vision_tune_to_meta,
                )

                state_dict[training.MODEL_KEY] = llama3_vision_tune_to_meta(
                    model_state_dict
                )
            else:
                # llama3_2 has tied weights, so we need to add the output.weight key
                if (
                    self._model_type == ModelType.LLAMA3_2
                    and "output.weight" not in model_state_dict
                ):
                    model_state_dict["output.weight"] = model_state_dict[
                        "tok_embeddings.weight"
                    ]

                state_dict[training.MODEL_KEY] = convert_weights.tune_to_meta(
                    model_state_dict
                )

            model_file_name = Path.joinpath(model_file_path, "consolidated.00.pth")

            torch.save(state_dict[training.MODEL_KEY], model_file_name)
            logger.info(
                "Model checkpoint of size "
                f"{os.path.getsize(model_file_name) / 1000**3:.2f} GB "
                f"saved to {model_file_name}"
            )

        if training.ADAPTER_KEY in state_dict:
            adapter_file_path = model_file_path / "adapter"
            adapter_file_path.mkdir(parents=True, exist_ok=True)
            adapter_file_name = Path.joinpath(adapter_file_path, "adapter.pth")
            torch.save(state_dict[training.ADAPTER_KEY], adapter_file_name)
            logger.info(
                "Adapter checkpoint of size "
                f"{os.path.getsize(adapter_file_name) / 1000**3:.2f} GB "
                f"saved to {adapter_file_name}"
            )

        elif adapter_only:
            raise ValueError(
                "Adapter checkpoint not found in state_dict. Please ensure that the state_dict contains adapter weights."
            )

        print("model_file_path", str(model_file_path))

        return str(model_file_path)
