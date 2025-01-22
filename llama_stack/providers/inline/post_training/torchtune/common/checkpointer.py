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
from torchtune.training.checkpointing._utils import (
    ADAPTER_CONFIG_FNAME,
    ADAPTER_MODEL_FNAME,
    check_outdir_not_in_ckptdir,
    copy_files,
    get_adapter_checkpoint_path,
    get_model_checkpoint_path,
    get_recipe_checkpoint_path,
    ModelType,
    RECIPE_STATE_DIRNAME,
    REPO_ID_FNAME,
    safe_torch_load,
    SAFETENSOR_INDEX_FNAME,
    SHARD_FNAME,
    SUFFIXES_TO_NOT_COPY,
    TORCH_INDEX_FNAME,
)
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
        checkpoint_format: str = "meta",
    ) -> str:
        model_file_path = (
            Path(self._output_dir)
            / f"{self._model_id}-{self._training_algorithm}-{epoch}"
        )
        if format == "meta":
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
        elif format == "hf":
            # the config.json file contains model params needed for state dict conversion
            config = json.loads(
                Path.joinpath(self._checkpoint_dir, "config.json").read_text()
            )
            if not adapter_only:
                state_dict[training.MODEL_KEY] = convert_weights.tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=config["num_attention_heads"],
                    num_kv_heads=config["num_key_value_heads"],
                    dim=config["hidden_size"],
                    head_dim=config.get("head_dim", None),
                )

                # split the state_dict into separate dicts, one for each output checkpoint file
                # e.g. split_state_dicts= {
                #       "0001": {"key1": tensor1, "key2": tensor2},
                #       "0002": {"key3": tensor3}
                #       }
                split_state_dicts: Dict[str, Dict[str, torch.Tensor]] = {}
                total_size = 0
                for key, weight in state_dict[training.MODEL_KEY].items():
                    cpt_idx = self._weight_map[key]

                    # initialize dict
                    if cpt_idx not in split_state_dicts:
                        split_state_dicts[cpt_idx] = {}

                    split_state_dicts[cpt_idx].update({key: weight})
                    total_size += weight.numel() * weight.element_size()

                # write the partitioned state dicts to the right checkpoint file
                # e.g. model-00001-of-00004.safetensors, model-00002-of-00004.safetensors, etc
                num_shards = len(split_state_dicts)
                map_original_name_to_new_name = {}
                for cpt_idx, model_state_dict in split_state_dicts.items():
                    # TODO: We should probably use the original shard name and just add a prefix
                    # however, having the SHARD_FNAME standardizes our checkpoints
                    shard_name = SHARD_FNAME.format(
                        cpt_idx=f"{cpt_idx}".zfill(5),
                        num_shards=f"{num_shards}".zfill(5),
                    )
                    map_original_name_to_new_name[cpt_idx] = shard_name
                    output_path = Path.joinpath(model_file_path, shard_name)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path = output_path.with_suffix(".safetensors")
                    save_file(model_state_dict, output_path, metadata={"format": "pt"})

                    logger.info(
                        "Model checkpoint of size "
                        f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
                        f"saved to {output_path}"
                    )

                # Save the appropriate index file based on serialization format
                # e.g. {metadata: {total_size: 1234}, weight_map: {"key1": "model_0001.safetensors", "key2": "model_0002.safetensors"}}
                weight_map = {
                    k: map_original_name_to_new_name[cpt_idx] + ".safetensors"
                    for k, cpt_idx in self._weight_map.items()
                }
                index_file_name = SAFETENSOR_INDEX_FNAME

                index_path = Path.joinpath(model_file_path, index_file_name)

                index_data = {
                    "metadata": {"total_size": total_size},
                    "weight_map": weight_map,
                }
                with open(index_path, "w") as f:
                    json.dump(index_data, f, indent=2)

            if training.ADAPTER_KEY in state_dict:

                # TODO: saving it "as is" is a requirement because, if we only save with
                # convert_weights.tune_to_peft_adapter_weights, we do NOT have a fn
                # convert_weights.peft_to_tune. The .pt format is not needed, but
                # it is an easy way to distinguish the adapters. Ideally we should save only one.
                output_path = Path.joinpath(
                    model_file_path, ADAPTER_MODEL_FNAME
                ).with_suffix(".pt")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(state_dict[training.ADAPTER_KEY], output_path)
                logger.info(
                    "Adapter checkpoint of size "
                    f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
                    f"saved to {output_path}"
                )

                state_dict[training.ADAPTER_KEY] = (
                    convert_weights.tune_to_peft_adapter_weights(
                        state_dict[training.ADAPTER_KEY],
                        num_heads=config["num_attention_heads"],
                        num_kv_heads=config["num_key_value_heads"],
                        dim=config["hidden_size"],
                        head_dim=config.get("head_dim", None),
                    )
                )
                output_path = Path.joinpath(model_file_path, ADAPTER_MODEL_FNAME)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path = output_path.with_suffix(".safetensors")
                save_file(
                    state_dict[training.ADAPTER_KEY],
                    output_path,
                    metadata={"format": "pt"},
                )
                logger.info(
                    "Adapter checkpoint of size "
                    f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
                    f"saved to {output_path}"
                )
            elif adapter_only:
                raise ValueError(
                    "Adapter checkpoint not found in state_dict. Please ensure that the state_dict contains adapter weights."
                )

            if training.ADAPTER_CONFIG in state_dict:
                state_dict[training.ADAPTER_CONFIG] = (
                    convert_weights.tune_to_peft_adapter_config(
                        adapter_config=state_dict[training.ADAPTER_CONFIG],
                        base_model_name_or_path=self.repo_id,
                    )
                )

                output_path = Path.joinpath(
                    model_file_path, ADAPTER_CONFIG_FNAME
                ).with_suffix(".json")
                with open(output_path, "w") as f:
                    json.dump(state_dict[training.ADAPTER_CONFIG], f)
                logger.info(
                    "Adapter checkpoint of size "
                    f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
                    f"saved to {output_path}"
                )

            # Save all files in ckpt_dir, except model weights and mapping, to output_dir/epoch_{epoch}
            # So its easy to run inference with the model using this epoch's checkpoint
            copy_files(
                self._checkpoint_dir,
                model_file_path,
                ignore_suffixes=SUFFIXES_TO_NOT_COPY,
            )
            logger.info("Saving final epoch checkpoint.")
            if adapter_only:
                logger.info(
                    "Please note that you have set adapter_only=True, so only adapter weights will be saved."
                    "You need to merge the adapter weights into your base model for further use. "
                    f"See {self.__class__.__name__}.save_checkpoint for more details."
                )
            else:
                logger.info(
                    "The full model checkpoint, including all weights and configurations, has been saved successfully."
                    "You can now use this checkpoint for further training or inference."
                )
        else:
            raise ValueError(f"Unsupported checkpoint format: {format}")

        return str(model_file_path)
