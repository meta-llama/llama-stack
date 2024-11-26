# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import logging
import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from llama_stack.providers.inline.post_training.meta_reference import utils
from llama_stack.providers.inline.post_training.meta_reference.configs import (
    MetaReferencePostTrainingConfig,
)
from llama_stack.providers.inline.post_training.meta_reference.datasets.sft import (
    SFTDataset,
)
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import modules, training
from torchtune.data import InputOutputToMessages, padded_collate_sft

from torchtune.modules.loss import CEWithChunkedOutputLoss
from torchtune.modules.peft import (
    get_adapter_params,
    get_adapter_state_dict,
    get_lora_module_names,
    get_merged_lora_ckpt,
    load_dora_magnitudes,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
)
from torchtune.training.lr_scheduler import get_cosine_schedule_with_warmup

log = logging.getLogger(__name__)

Tokenizer = Union[Llama3Tokenizer]


class LoraFinetuningSingleDevice:
    def __init__(
        self,
        config: MetaReferencePostTrainingConfig,
        request: PostTrainingSFTRequest,
        datasetio_api: DatasetIOAPI,
    ) -> None:
        # to make user config easier, assume the device is 'cuda' only
        # self._device = utils.get_device(device=cfg.device)
        self.config = config
        self.request = request
        self._device = "cuda"
        self._dtype = training.get_dtype(
            request.training_config.dtype, device=self._device
        )
        self.model_id = request.model

        # hardcode it for now and see how it works with get_training_job_artifacts
        self._output_dir = f"~/.llama/checkpoints/post_training/{request.model_id}"

        self._log_every_n_steps = 1
        self._log_peak_memory_stats = False

        self.seed = training.set_seed(seed=config.torch_seed or 42)
        self.epochs_run = 0
        self.total_epochs = request.training_config.n_epochs
        self._shuffle = request.training_config.shuffle
        self._batch_size = request.training_config.batch_size

        # this is important for debugging purpose
        self.max_steps_per_epoch = request.training_config.max_steps_per_epoch
        self.global_step = 0

        # not needed in MVP
        # self._resume_from_checkpoint = cfg.resume_from_checkpoint
        # self._save_adapter_weights_only = cfg.get("save_adapter_weights_only", False)

        self._gradient_accumulation_steps = (
            request.training_config.gradient_accumulation_steps
        )

        self._clip_grad_norm = 1.0  # hardcode
        self._enable_activation_checkpointing = (
            request.training_config.enable_activation_checkpointing
        )
        self._enable_activation_offloading = False

        self.datasetio_api = datasetio_api

    def load_checkpoint(self):
        def get_checkpoint_files(checkpoint_dir: str) -> List[str]:
            try:
                # List all files in the given directory
                files = os.listdir(checkpoint_dir)

                # Filter files that end with .pth
                pth_files = [file for file in files if file.endswith(".pth")]
                return pth_files
            except FileNotFoundError:
                return f"Error: The directory '{checkpoint_dir}' does not exist."

        self._checkpointer = training.FullModelMetaCheckpointer(
            checkpoint_dir=self.config.checkpoint_dir,
            checkpoint_files=get_checkpoint_files,
            output_dir=self._output_dir,
            # todo: automatically get this info from model
            model_type="LLAMA3",
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()
        return checkpoint_dict

    def setup(self, config: MetaReferencePostTrainingConfig) -> None:
        # todo: figure out how does it works with telemetry
        # self._metric_logger = config.instantiate(cfg.metric_logger)
        # self._metric_logger.log_config(cfg)

        checkpoint_dict = self.load_checkpoint()

        # hack to toggle to the low cpu ram version of the reparametrize_as_dtype
        # hook based on the config.
        # common_utils._use_low_cpu_ram = cfg.get("low_cpu_ram", False)

        # set up model
        self._model = self._setup_model(
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            base_model_state_dict=checkpoint_dict[training.MODEL_KEY],
            lora_weights_state_dict=None,
        )

        self._tokenizer = self._setup_tokenizer()
        log.info("Tokenizer is initialized from file.")

        self._optimizer = self._setup_optimizer(
            optimizer_config=self.request.training_config.optimizer, opt_state_dict=None
        )

        self._loss_fn = CEWithChunkedOutputLoss()
        self._sampler, self._dataloader = self._setup_data(
            tokenizer=self._tokenizer,
            shuffle=self._shuffle,
            batch_size=self._batch_size,
        )

        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader and the max_steps_per_epoch param set by the user and is used
        # for logging and tracking training state. This should be computed after the dataloader
        # has been setup
        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
            self.global_step = self.epochs_run * self._steps_per_epoch

        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        self._lr_scheduler = self._setup_lr_scheduler(
            num_warmup_steps=self.request.optimizer_config.num_warmup_steps,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        # Used to ignore labels for loss computation
        self.ignore_labels_cache = torch.full(
            (self._batch_size, 1), self._loss_fn.ignore_index, device=self._device
        )

    def _setup_model(
        self,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        base_model_state_dict: Dict[str, Any],
        lora_weights_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        self._lora_rank = self.request.algorithm_config.rank
        self._lora_alpha = self.request.algorithm_config.alpha
        self._lora_attn_modules = list(self.request.algorithm_config.lora_attn_modules)
        self._apply_lora_to_mlp = self.request.algorithm_config.apply_lora_to_mlp
        self._apply_lora_to_output = self.request.algorithm_config.apply_lora_to_output
        self._use_dora = self.request.algorithm_config.use_dora

        with training.set_default_dtype(self._dtype), self._device:
            model_type = utils.get_model_type(self.model_id)
            model = model_type(
                lora_attn_modules=self._lora_attn_modules,
                apply_lora_to_mlp=self._apply_lora_to_mlp,
                apply_lora_to_output=self._apply_lora_to_output,
                lora_rank=self._lora_rank,
                lora_alpha=self._lora_alpha,
                quantize_base=False,
                use_dora=self._use_dora,
            )

        self.adapter_params = get_adapter_params(model)
        self._is_dora = any(["magnitude" in k for k in self.adapter_params.keys()])

        set_trainable_params(model, self.adapter_params)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        base_missing, base_unexpected = model.load_state_dict(
            base_model_state_dict, strict=False
        )

        # This is for any adapters that need to be initialized after base weights
        # have been loaded (e.g. DoRA).
        if self._is_dora:
            for m in model.modules():
                if hasattr(m, "initialize_dora_magnitude"):
                    m.initialize_dora_magnitude()
            load_dora_magnitudes(model)
        if lora_weights_state_dict:
            lora_missing, lora_unexpected = model.load_state_dict(
                lora_weights_state_dict, strict=False
            )
        else:
            lora_missing, lora_unexpected = None, None
        validate_missing_and_unexpected_for_lora(
            lora_attn_modules=self._lora_attn_modules,
            apply_lora_to_mlp=self._apply_lora_to_mlp,
            apply_lora_to_output=self._apply_lora_to_output,
            base_missing=base_missing,
            base_unexpected=base_unexpected,
            lora_missing=lora_missing,
            lora_unexpected=lora_unexpected,
        )
        # Validate model adapter params were loaded in with the expected dtype
        # TODO (rohan-varma): Further validation to ensure the appropriate base params
        # are NF4 vs bf16 based on the quantization config.
        training.validate_expected_param_dtype(
            self.adapter_params.items(), dtype=self._dtype
        )

        # activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )

        log.info(f"Model is initialized with precision {self._dtype}.")

        # if self._device.type != "cpu":
        #     memory_stats = training.get_memory_stats(device=self._device)
        #     training.log_memory_stats(memory_stats)
        return model

    def _setup_tokenizer(
        self,
    ) -> Tokenizer:
        tokenizer_path = self.config.checkpoint_dir + "/tokenizer.model"
        tokenizer_type = utils.get_tokenizer_type(self.model_id)
        return tokenizer_type(path=tokenizer_path)

    def _setup_optimizer(self, optimizer_config: OptimizerConfig) -> Optimizer:
        optimizer = torch.optim.AdamW(
            params=self._model.parameters(),
            lr=optimizer_config.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
        )

        log.info("Optimizer and loss are initialized.")
        return optimizer

    async def _setup_data(
        self, tokenizer: Tokenizer, shuffle: bool, batch_size: int
    ) -> Tuple[DistributedSampler, DataLoader]:
        all_rows = await self.datasetio_api.get_rows_paginated(
            dataset_id=self.request.dataset_id,
            rows_in_page=-1,
        )
        rows = all_rows.rows

        ds = SFTDataset(
            rows, message_transform=InputOutputToMessages(), model_transform=tokenizer
        )

        sampler = DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            sampler=sampler,
            batch_size=batch_size,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
            collate_fn=(
                partial(
                    padded_collate_sft,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                )
            ),
        )

        log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def _setup_lr_scheduler(
        self,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        lr_scheduler = get_cosine_schedule_with_warmup(
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        log.info("Learning rate scheduler is initialized.")
        return lr_scheduler