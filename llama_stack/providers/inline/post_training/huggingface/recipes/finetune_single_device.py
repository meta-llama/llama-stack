# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import logging
import gc
import time
import psutil
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Set tokenizer parallelism environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.post_training import (
    Checkpoint,
    DataConfig,
    LoraFinetuningConfig,
    TrainingConfig,
)

from ..config import HuggingFacePostTrainingConfig

logger = logging.getLogger(__name__)

def get_memory_stats(device: torch.device) -> dict[str, Any]:
    """Get memory statistics for the given device."""
    stats = {
        "system_memory": {
            "total": psutil.virtual_memory().total / (1024**3),  # GB
            "available": psutil.virtual_memory().available / (1024**3),  # GB
            "used": psutil.virtual_memory().used / (1024**3),  # GB
            "percent": psutil.virtual_memory().percent,
        }
    }
    
    if device.type == "cuda":
        stats["device_memory"] = {
            "allocated": torch.cuda.memory_allocated(device) / (1024**3),  # GB
            "reserved": torch.cuda.memory_reserved(device) / (1024**3),    # GB
            "max_allocated": torch.cuda.max_memory_allocated(device) / (1024**3),  # GB
        }
    elif device.type == "mps":
        # MPS doesn't provide direct memory stats, but we can track system memory
        stats["device_memory"] = {
            "note": "MPS memory stats not directly available",
            "system_memory_used": psutil.virtual_memory().used / (1024**3),  # GB
        }
    elif device.type == "cpu":
        # For CPU, we track process memory usage
        process = psutil.Process()
        stats["device_memory"] = {
            "process_rss": process.memory_info().rss / (1024**3),  # GB
            "process_vms": process.memory_info().vms / (1024**3),  # GB
            "process_percent": process.memory_percent(),
        }
    
    return stats

class HFFinetuningSingleDevice:
    def __init__(
        self,
        job_uuid,
        datasetio_api: DatasetIO,
        datasets_api: Datasets,
    ):
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api
        self.job_uuid = job_uuid

    def validate_dataset_format(self, rows: list[dict]) -> bool:
        """Validate that the dataset has the required fields."""
        required_fields = ["input_query", "expected_answer", "chat_completion_input"]
        return all(field in row for row in rows for field in required_fields)

    def _process_instruct_format(self, row: dict) -> tuple[str | None, str | None]:
        """Process a row in instruct format."""
        if "chat_completion_input" in row and "expected_answer" in row:
            try:
                messages = json.loads(row["chat_completion_input"])
                if not isinstance(messages, list) or len(messages) != 1:
                    logger.warning(f"Invalid chat_completion_input format: {row['chat_completion_input']}")
                    return None, None
                if "content" not in messages[0]:
                    logger.warning(f"Message missing content: {messages[0]}")
                    return None, None
                return messages[0]["content"], row["expected_answer"]
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse chat_completion_input: {row['chat_completion_input']}")
                return None, None
        return None, None

    def _process_dialog_format(self, row: dict) -> tuple[str | None, str | None]:
        """Process a row in dialog format."""
        if "dialog" in row:
            try:
                dialog = json.loads(row["dialog"])
                if not isinstance(dialog, list) or len(dialog) < 2:
                    logger.warning(f"Dialog must have at least 2 messages: {row['dialog']}")
                    return None, None
                if dialog[0].get("role") != "user":
                    logger.warning(f"First message must be from user: {dialog[0]}")
                    return None, None
                if not any(msg.get("role") == "assistant" for msg in dialog):
                    logger.warning("Dialog must have at least one assistant message")
                    return None, None
                
                # Convert to human/gpt format
                role_map = {"user": "human", "assistant": "gpt"}
                conversations = []
                for msg in dialog:
                    if "role" not in msg or "content" not in msg:
                        logger.warning(f"Message missing role or content: {msg}")
                        continue
                    conversations.append({
                        "from": role_map[msg["role"]],
                        "value": msg["content"]
                    })
                
                # Format as a single conversation
                return conversations[0]["value"], conversations[1]["value"]
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse dialog: {row['dialog']}")
                return None, None
        return None, None

    def _process_fallback_format(self, row: dict) -> tuple[str | None, str | None]:
        """Process a row using fallback formats."""
        if "input" in row and "output" in row:
            return row["input"], row["output"]
        elif "prompt" in row and "completion" in row:
            return row["prompt"], row["completion"]
        elif "question" in row and "answer" in row:
            return row["question"], row["answer"]
        return None, None

    def _format_text(self, input_text: str, output_text: str, provider_config: HuggingFacePostTrainingConfig) -> str:
        """Format input and output text based on model requirements."""
        if hasattr(provider_config, 'chat_template'):
            return provider_config.chat_template.format(
                input=input_text,
                output=output_text
            )
        return f"{input_text}\n{output_text}"

    def _create_dataset(self, rows: list[dict], config: TrainingConfig, provider_config: HuggingFacePostTrainingConfig) -> Dataset:
        """Create and preprocess the dataset."""
        formatted_rows = []
        for row in rows:
            input_text = None
            output_text = None

            # Process based on format
            if config.data_config.data_format.value == "instruct":
                input_text, output_text = self._process_instruct_format(row)
            elif config.data_config.data_format.value == "dialog":
                input_text, output_text = self._process_dialog_format(row)
            else:
                input_text, output_text = self._process_fallback_format(row)

            if input_text and output_text:
                formatted_text = self._format_text(input_text, output_text, provider_config)
                formatted_rows.append({"text": formatted_text})

        if not formatted_rows:
            raise ValueError(f"No valid input/output pairs found in the dataset for format: {config.data_config.data_format.value}")

        return Dataset.from_list(formatted_rows)

    def _preprocess_dataset(self, ds: Dataset, tokenizer: AutoTokenizer, provider_config: HuggingFacePostTrainingConfig) -> Dataset:
        """Preprocess the dataset with tokenizer."""
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding=True,
                truncation=True,
                max_length=provider_config.max_seq_length,
                return_tensors=None,
            )
        
        return ds.map(
            tokenize_function,
            batched=True,
            remove_columns=ds.column_names,
        )

    async def train(
        self,
        model: str,
        output_dir: str,
        job_uuid: str,
        lora_config: LoraFinetuningConfig,
        config: TrainingConfig,
        provider_config: HuggingFacePostTrainingConfig,
    ) -> tuple[dict[str, Any], list[Checkpoint]]:
        """Train a model using HuggingFace's SFTTrainer"""
        try:
            device = torch.device(provider_config.device)
        except RuntimeError as e:
            raise RuntimeError(f"Error getting Torch Device {str(e)}") from e

        # Detect device type and validate
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"{device.type}: Torch has no CUDA/ROCm support or could not detect a compatible device."
                )
            # map unqualified 'cuda' to current device
            if device.index is None:
                device = torch.device(device.type, torch.cuda.current_device())
        elif device.type == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    f"{device.type}: Torch has no MPS support or could not detect a compatible device."
                )
        elif device.type == "hpu":
            raise RuntimeError(f"{device.type}: training does not support Intel Gaudi.")

        logger.info(f"Using device '{device}'")
        output_dir = Path(output_dir)

        # Track memory stats throughout training
        memory_stats = {
            "initial": get_memory_stats(device),
            "after_model_load": None,
            "after_training": None,
            "final": None
        }

        # Load dataset
        assert isinstance(config.data_config, DataConfig), "DataConfig must be initialized"
        rows = await self._setup_data(config.data_config.dataset_id)

        # Validate that the dataset has the required fields for training
        if not self.validate_dataset_format(rows):
            raise ValueError("Dataset is missing required fields: input_query, expected_answer, chat_completion_input")

        # Initialize tokenizer with model-specific config
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                **provider_config.model_specific_config
            )
            # Set up tokenizer defaults
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            tokenizer.truncation_side = "right"
            tokenizer.model_max_length = provider_config.max_seq_length
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tokenizer: {str(e)}")

        # Create and preprocess dataset
        try:
            ds = self._create_dataset(rows, config, provider_config)
            ds = self._preprocess_dataset(ds, tokenizer, provider_config)
        except Exception as e:
            raise ValueError(f"Failed to create dataset: {str(e)}")

        # Load model with model-specific config
        logger.info("Loading the base model")
        try:
            model_config = AutoConfig.from_pretrained(
                model,
                **provider_config.model_specific_config
            )
            model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype="auto",
                quantization_config=None,
                config=model_config,
                **provider_config.model_specific_config
            )
            if model.device != device:
                model = model.to(device)
            logger.info(f"Model device {model.device}")
            memory_stats["after_model_load"] = get_memory_stats(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

        # Configure LoRA
        peft_config = LoraConfig(
            lora_alpha=lora_config.alpha,
            lora_dropout=0.1,
            r=lora_config.rank,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_config.lora_attn_modules,
        )

        # Setup training arguments
        training_arguments = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=config.n_epochs,
            per_device_train_batch_size=config.data_config.batch_size,
            fp16=device.type == "cuda",
            bf16=device.type != "cuda",
            use_cpu=device.type == "cpu",
            save_strategy="epoch",
            report_to="none",
            max_seq_length=provider_config.max_seq_length,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            gradient_checkpointing=provider_config.gradient_checkpointing,
            learning_rate=config.optimizer_config.lr if hasattr(config.optimizer_config, 'lr') else 2e-5,
            warmup_ratio=provider_config.warmup_ratio,
            weight_decay=provider_config.weight_decay,
            logging_steps=provider_config.logging_steps,
            eval_strategy="no",
            save_total_limit=provider_config.save_total_limit,
            remove_unused_columns=False,
            dataloader_pin_memory=provider_config.dataloader_pin_memory,
            dataloader_num_workers=provider_config.dataloader_num_workers,
            dataset_text_field="text",
            packing=False,
        )

        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=ds,
            peft_config=peft_config,
            args=training_arguments,
        )

        # Train
        logger.info("Starting training")
        try:
            trainer.train()
            memory_stats["after_training"] = get_memory_stats(device)

            # Save final model
            model.config.use_cache = True
            model = trainer.model.merge_and_unload()
            model.save_pretrained(output_dir / "merged_model")

            # Create checkpoint
            checkpoint = Checkpoint(
                identifier=f"{model}-sft-{config.n_epochs}",
                created_at=datetime.now(timezone.utc),
                epoch=config.n_epochs,
                post_training_job_id=job_uuid,
                path=str(output_dir / "merged_model"),
            )

            return memory_stats, [checkpoint]
        finally:
            # Clean up resources
            if hasattr(trainer, 'model'):
                if device.type != "cpu":
                    trainer.model.to("cpu")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                del trainer.model
            del trainer
            gc.collect()
            memory_stats["final"] = get_memory_stats(device)

    async def _setup_data(
        self,
        dataset_id: str,
    ) -> list[dict[str, Any]]:
        """Load dataset from llama stack dataset provider"""
        try:
            async def fetch_rows(dataset_id: str):
                return await self.datasetio_api.iterrows(
                    dataset_id=dataset_id,
                    limit=-1,
                )

            all_rows = await fetch_rows(dataset_id)
            return all_rows.data
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {str(e)}")
