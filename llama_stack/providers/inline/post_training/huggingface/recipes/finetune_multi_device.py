# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import gc
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.distributed as dist
from peft import LoraConfig, LoraModel
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel, StateDictType

from llama_stack.providers.inline.post_training.common.utils import evacuate_model_from_device

# Set tokenizer parallelism environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Force PyTorch to use OpenBLAS instead of MKL
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "0"
os.environ["MKL_NUM_THREADS"] = "1"


def configure_nccl_logging(
    debug: bool = False,
    debug_subsys: str = "NONE",
    socket_timeout: int = 1200,
    async_error_handling: bool = True,
    blocking_wait: bool = True,
    ib_timeout: int = 120,
    net_gdr_level: int = 5,
    cuda_launch_blocking: bool = True,
) -> None:
    """Configure NCCL environment variables for distributed training.
    Args:
        debug: Enable NCCL debug logging
        debug_subsys: NCCL subsystems to debug (ALL, INIT, COLL, P2P, SHM, NET, etc.)
        socket_timeout: Socket timeout in seconds
        async_error_handling: Enable async error handling
        blocking_wait: Use blocking wait
        ib_timeout: InfiniBand timeout in seconds
        net_gdr_level: GPU Direct RDMA level (0-5)
        cuda_launch_blocking: Enable CUDA launch blocking
    """
    # Set NCCL environment variables
    os.environ["NCCL_DEBUG"] = "INFO" if debug else "WARN"
    os.environ["NCCL_DEBUG_SUBSYS"] = debug_subsys
    os.environ["NCCL_SOCKET_TIMEOUT"] = str(socket_timeout)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1" if async_error_handling else "0"
    os.environ["NCCL_BLOCKING_WAIT"] = "1" if blocking_wait else "0"
    os.environ["NCCL_IB_TIMEOUT"] = str(ib_timeout)
    os.environ["NCCL_NET_GDR_LEVEL"] = str(net_gdr_level)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" if cuda_launch_blocking else "0"


# Configure NCCL with default settings (minimal logging)
configure_nccl_logging()

from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from trl import SFTConfig, SFTTrainer

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


def get_gb(to_convert: int) -> str:
    """Converts memory stats to GB and formats to 2 decimal places.

    Args:
        to_convert: Memory value in bytes

    Returns:
        str: Memory value in GB formatted to 2 decimal places
    """
    return f"{(to_convert / (1024**3)):.2f}"


def get_memory_stats(device: torch.device) -> dict[str, Any]:
    """Get memory statistics for the given device.

    This function collects memory statistics for both system and device memory.
    For CUDA devices, it tracks allocated, reserved, and max allocated memory.
    For MPS devices, it tracks system memory usage since direct device stats aren't available.
    For CPU devices, it tracks process memory usage.

    Args:
        device: The device to get memory stats for (cuda, mps, or cpu)

    Returns:
        dict: Dictionary containing memory statistics for both system and device
    """
    stats = {
        "system_memory": {
            "total": get_gb(psutil.virtual_memory().total),
            "available": get_gb(psutil.virtual_memory().available),
            "used": get_gb(psutil.virtual_memory().used),
            "percent": psutil.virtual_memory().percent,
        }
    }

    if device.type == "cuda":
        stats["device_memory"] = {
            "allocated": get_gb(torch.cuda.memory_allocated(device)),
            "reserved": get_gb(torch.cuda.memory_reserved(device)),
            "max_allocated": get_gb(torch.cuda.max_memory_allocated(device)),
        }
    elif device.type == "mps":
        # MPS doesn't provide direct memory stats, but we can track system memory
        stats["device_memory"] = {
            "note": "MPS memory stats not directly available",
            "system_memory_used": get_gb(psutil.virtual_memory().used),
        }
    elif device.type == "cpu":
        # For CPU, we track process memory usage
        process = psutil.Process()
        stats["device_memory"] = {
            "process_rss": get_gb(process.memory_info().rss),
            "process_vms": get_gb(process.memory_info().vms),
            "process_percent": process.memory_percent(),
        }

    return stats


def setup_distributed_training(device_str: str) -> tuple[int, int]:
    """Initialize distributed training environment.

    This function sets up the distributed training environment by:
    1. Parsing the device list to determine number of GPUs
    2. Setting up environment variables for distributed training
    3. Initializing the process group with NCCL backend

    Args:
        device_str: Comma-separated list of devices (e.g. "cuda:0,cuda:1")

    Returns:
        tuple: (local_rank, world_size) where:
            - local_rank is the rank of this process (0 for single device)
            - world_size is the total number of processes (1 for single device)
    """
    # Parse device list
    devices = [d.strip() for d in device_str.split(",")]
    world_size = len(devices)

    if world_size <= 1:
        logger.info("Single device training")
        return 0, 1

    # Set up environment variables for distributed training
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = "0"  # We're the main process
    os.environ["LOCAL_RANK"] = "0"

    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    logger.info(f"Initialized distributed training with {world_size} devices: {devices}")

    return 0, world_size


class HFFinetuningMultiDevice:
    def __init__(
        self,
        job_uuid: str,
        datasetio_api: DatasetIO,
        datasets_api: Datasets,
        enable_nccl_debug: bool = False,
        nccl_debug_subsys: str = "NONE",
    ):
        """Initialize the multi-device fine-tuning handler.

        Args:
            job_uuid: Unique identifier for this training job
            datasetio_api: API for dataset I/O operations
            datasets_api: API for dataset management
        """
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api
        self.job_uuid = job_uuid
        self.enable_nccl_debug = enable_nccl_debug
        self.nccl_debug_subsys = nccl_debug_subsys

    def validate_dataset_format(self, rows: list[dict]) -> bool:
        """Validate that the dataset has the required fields.

        Args:
            rows: List of dataset rows to validate

        Returns:
            bool: True if all rows have required fields, False otherwise
        """
        required_fields = ["input_query", "expected_answer", "chat_completion_input"]
        return all(field in row for row in rows for field in required_fields)

    def _process_instruct_format(self, row: dict) -> tuple[str | None, str | None]:
        """Process a row in instruct format.

        Args:
            row: Dataset row containing chat completion input and expected answer

        Returns:
            tuple: (input_text, output_text) or (None, None) if invalid format
        """
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
        """Process a row in dialog format.

        Args:
            row: Dataset row containing dialog messages

        Returns:
            tuple: (input_text, output_text) or (None, None) if invalid format
        """
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
                    conversations.append({"from": role_map[msg["role"]], "value": msg["content"]})

                # Format as a single conversation
                return conversations[0]["value"], conversations[1]["value"]
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse dialog: {row['dialog']}")
                return None, None
        return None, None

    def _process_fallback_format(self, row: dict) -> tuple[str | None, str | None]:
        """Process a row using fallback formats.

        Args:
            row: Dataset row to process

        Returns:
            tuple: (input_text, output_text) or (None, None) if no valid format found
        """
        if "input" in row and "output" in row:
            return row["input"], row["output"]
        elif "prompt" in row and "completion" in row:
            return row["prompt"], row["completion"]
        elif "question" in row and "answer" in row:
            return row["question"], row["answer"]
        return None, None

    def _format_text(self, input_text: str, output_text: str, provider_config: HuggingFacePostTrainingConfig) -> str:
        """Format input and output text based on model requirements.

        Args:
            input_text: The input text to format
            output_text: The output text to format
            provider_config: Configuration containing chat template

        Returns:
            str: Formatted text using the chat template
        """
        if hasattr(provider_config, "chat_template"):
            return provider_config.chat_template.format(input=input_text, output=output_text)
        return f"{input_text}\n{output_text}"

    def _create_dataset(
        self, rows: list[dict], config: TrainingConfig, provider_config: HuggingFacePostTrainingConfig
    ) -> Dataset:
        """Create and preprocess the dataset.

        Args:
            rows: List of dataset rows to process
            config: Training configuration containing data format
            provider_config: Provider-specific configuration

        Returns:
            Dataset: Processed dataset ready for training

        Raises:
            ValueError: If no valid input/output pairs found for the specified format
        """
        formatted_rows = []
        for row in rows:
            input_text = None
            output_text = None

            # Process based on format
            assert isinstance(config.data_config, DataConfig), "DataConfig must be initialized"
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
            assert isinstance(config.data_config, DataConfig), "DataConfig must be initialized"
            raise ValueError(
                f"No valid input/output pairs found in the dataset for format: {config.data_config.data_format.value}"
            )

        return Dataset.from_list(formatted_rows)

    def _preprocess_dataset(
        self, ds: Dataset, tokenizer: AutoTokenizer, provider_config: HuggingFacePostTrainingConfig
    ) -> Dataset:
        """Preprocess the dataset with tokenizer.

        Args:
            ds: Dataset to preprocess
            tokenizer: Tokenizer to use for preprocessing
            provider_config: Provider-specific configuration

        Returns:
            Dataset: Tokenized and preprocessed dataset
        """

        def tokenize_function(examples):
            # Ensure consistent padding and truncation
            outputs = tokenizer(
                examples["text"],
                padding="max_length",  # Use max_length padding for consistent dimensions
                truncation=True,
                max_length=provider_config.max_seq_length,
                return_tensors=None,  # Don't return tensors yet
                return_attention_mask=True,
                return_token_type_ids=False,
            )
            # Add labels for causal language modeling
            outputs["labels"] = outputs["input_ids"].copy()

            # Verify dimensions
            assert all(len(x) == provider_config.max_seq_length for x in outputs["input_ids"]), (
                "Inconsistent input_ids length"
            )
            assert all(len(x) == provider_config.max_seq_length for x in outputs["attention_mask"]), (
                "Inconsistent attention_mask length"
            )
            assert all(len(x) == provider_config.max_seq_length for x in outputs["labels"]), (
                "Inconsistent labels length"
            )

            return outputs

        # Process in batches
        return ds.map(
            tokenize_function,
            batched=True,
            batch_size=1000,  # Process in larger batches for efficiency
            remove_columns=ds.column_names,
            desc="Tokenizing and preparing dataset",
            num_proc=1,  # Single process to avoid issues
        )

    async def _setup_data(self, dataset_id: str) -> list[dict[str, Any]]:
        """Load dataset from llama stack dataset provider.

        Args:
            dataset_id: ID of the dataset to load

        Returns:
            list: List of dataset rows

        Raises:
            RuntimeError: If dataset loading fails
        """
        try:
            all_rows = await self.datasetio_api.iterrows(
                dataset_id=dataset_id,
                limit=-1,
            )
            if not isinstance(all_rows.data, list):
                raise RuntimeError("Expected dataset data to be a list")
            return all_rows.data
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {str(e)}") from e

    def _run_training_sync(
        self,
        local_rank: int,  # First parameter must be local_rank for spawn
        world_size: int,  # Second parameter must be world_size for spawn
        model: str,
        provider_config: dict[str, Any],
        peft_config: LoraConfig | None,
        config: dict[str, Any],
        output_dir_path: Path | None,
    ) -> None:
        """Synchronous wrapper for running training process.

        This method serves as a bridge between the multiprocessing Process and the async training function.
        It creates a new event loop to run the async training process.

        Args:
            local_rank: Local rank of this process (0 to world_size-1)
            world_size: Total number of processes
            model: The model identifier to load
            provider_config: Configuration specific to the HuggingFace provider
            peft_config: Optional LoRA configuration
            config: General training configuration
            output_dir_path: Optional path to save the model
        """
        import asyncio

        logger.info("Starting training process with async wrapper")
        asyncio.run(
            self._run_training(
                model=model,
                provider_config=provider_config,
                peft_config=peft_config,
                config=config,
                output_dir_path=output_dir_path,
                local_rank=local_rank,
                world_size=world_size,
            )
        )

    async def _run_training(
        self,
        model: str,
        provider_config: dict[str, Any],
        peft_config: LoraConfig | None,
        config: dict[str, Any],
        output_dir_path: Path | None,
        local_rank: int,
        world_size: int,
    ) -> None:
        """Run the training process with signal handling.

        This method handles the actual training process, including:
        1. Setting up signal handlers for graceful shutdown
        2. Initializing distributed training environment
        3. Loading and preprocessing the dataset
        4. Loading and configuring the model
        5. Setting up and running the trainer
        6. Saving the final model
        7. Cleaning up resources

        Args:
            model: The model identifier to load
            provider_config: Configuration specific to the HuggingFace provider
            peft_config: Optional LoRA configuration
            config: General training configuration
            output_dir_path: Optional path to save the model
            local_rank: Local rank of this process
            world_size: Total number of processes
        """

        def signal_handler(signum, frame):
            """Handle termination signals gracefully."""
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Convert config dicts back to objects
        logger.info("Initializing configuration objects")
        provider_config_obj = HuggingFacePostTrainingConfig(**provider_config)
        config_obj = TrainingConfig(**config)

        # Set device for this process first
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        logger.info(f"Process {local_rank} using device {device}")

        # Set environment variables for this process
        # These are used by PyTorch's distributed module to coordinate between processes
        os.environ["LOCAL_RANK"] = str(local_rank)  # Unique rank for this process
        os.environ["RANK"] = str(local_rank)  # Global rank (same as local in our case)
        os.environ["WORLD_SIZE"] = str(world_size)  # Total number of processes
        os.environ["MASTER_ADDR"] = "localhost"  # Address of the main process
        os.environ["MASTER_PORT"] = "29500"  # Port for process communication

        # Initialize process group with NCCL backend
        # NCCL is NVIDIA's library for multi-GPU communication
        # This must be called after setting environment variables and device
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=local_rank,
            )
        logger.info(f"Initialized process group for rank {local_rank}")
        dist.barrier()

        # Load dataset and tokenizer
        train_dataset, eval_dataset, tokenizer = await self.load_dataset(model, config_obj, provider_config_obj)

        # Calculate steps per epoch
        if not config_obj.data_config:
            raise ValueError("DataConfig is required for training")
        steps_per_epoch = len(train_dataset) // config_obj.data_config.batch_size

        # Load model
        logger.info("Loading the base model")
        model_obj = self.load_model(model, device, provider_config_obj, peft_config)
        dist.barrier()

        # Setup training arguments
        training_args = self.setup_training_args(
            config_obj,
            provider_config_obj,
            output_dir_path,
            peft_config,
            steps_per_epoch,
            model_obj,
        )

        # Initialize trainer
        logger.info("Initializing SFTTrainer")
        trainer = SFTTrainer(
            model=model_obj,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            args=training_args,
        )

        try:
            # Train
            logger.info("Starting training")
            trainer.train()
            logger.info("Training completed successfully")

            # Save final model if output directory is provided
            if output_dir_path:  # and local_rank == 0:
                trainer.save_model(output_dir=output_dir_path)
                # self.save_model(local_rank, model, model_obj, tokenizer, trainer, peft_config, output_dir_path)

        finally:
            # Clean up resources
            logger.info("Cleaning up resources")
            if hasattr(trainer, "model"):
                evacuate_model_from_device(trainer.model, device.type)
            del trainer
            dist.barrier()
            dist.destroy_process_group()
            gc.collect()
            logger.info("Cleanup completed")

    async def load_dataset(
        self,
        model: str,
        config: TrainingConfig,
        provider_config: HuggingFacePostTrainingConfig,
    ) -> tuple[Dataset, Dataset, AutoTokenizer]:
        """Load and prepare the dataset for training.

        This method:
        1. Loads the dataset from the dataset provider
        2. Initializes the tokenizer for the model
        3. Creates and preprocesses the dataset
        4. Splits the dataset into train and validation sets

        Args:
            model: The model identifier to load
            config: Training configuration
            provider_config: Provider-specific configuration

        Returns:
            tuple: (train_dataset, eval_dataset, tokenizer)

        Raises:
            ValueError: If dataset is missing required fields
            RuntimeError: If dataset loading or tokenizer initialization fails
        """
        # Validate data config
        if not config.data_config:
            raise ValueError("DataConfig is required for training")

        # Load dataset
        logger.info(f"Loading dataset: {config.data_config.dataset_id}")
        rows = await self._setup_data(config.data_config.dataset_id)
        if not self.validate_dataset_format(rows):
            raise ValueError("Dataset is missing required fields: input_query, expected_answer, chat_completion_input")
        logger.info(f"Loaded {len(rows)} rows from dataset")

        # Initialize tokenizer
        logger.info(f"Initializing tokenizer for model: {model}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model, **provider_config.model_specific_config)

            # Set pad token to eos token if not present
            # This is common for models that don't have a dedicated pad token
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token

            # Set padding side to right for causal language modeling
            # This ensures that padding tokens don't interfere with the model's ability
            # to predict the next token in the sequence
            tokenizer.padding_side = "right"

            # Set truncation side to right to keep the beginning of the sequence
            # This is important for maintaining context and instruction format
            tokenizer.truncation_side = "right"

            # Set model max length to match provider config
            # This ensures consistent sequence lengths across the training process
            tokenizer.model_max_length = provider_config.max_seq_length

            logger.info("Tokenizer initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tokenizer: {str(e)}") from e

        # Create and preprocess dataset
        logger.info("Creating and preprocessing dataset")
        try:
            ds = self._create_dataset(rows, config, provider_config)
            ds = self._preprocess_dataset(ds, tokenizer, provider_config)
            logger.info(f"Dataset created with {len(ds)} examples")
        except Exception as e:
            raise ValueError(f"Failed to create dataset: {str(e)}") from e

        # Split dataset
        logger.info("Splitting dataset into train and validation sets")
        train_val_split = ds.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_val_split["train"]
        eval_dataset = train_val_split["test"]
        logger.info(f"Split dataset into {len(train_dataset)} training and {len(eval_dataset)} validation examples")

        return train_dataset, eval_dataset, tokenizer

    def load_model(
        self,
        model: str,
        device: torch.device,
        provider_config: HuggingFacePostTrainingConfig,
        peft_config: LoraConfig | None = None,
    ) -> AutoModelForCausalLM:
        """Load and initialize the model for training.

        This method:
        1. Loads the model configuration
        2. Determines optimal dtype based on device capabilities
        3. Loads the model with specified dtype
        4. Applies LoRA if configured
        5. Moves the model to the specified device

        Args:
            model: The model identifier to load
            device: The device to load the model on
            provider_config: Provider-specific configuration
            peft_config: Optional LoRA configuration

        Returns:
            AutoModelForCausalLM: The loaded and configured model

        Raises:
            RuntimeError: If model loading fails
        """
        logger.info("Loading the base model")
        try:
            model_config = AutoConfig.from_pretrained(model, **provider_config.model_specific_config)

            # Determine optimal dtype based on device capabilities
            if device.type == "cuda":
                if torch.cuda.is_bf16_supported():
                    torch_dtype = torch.bfloat16
                    logger.info("Using bfloat16 precision (supported by device)")
                else:
                    torch_dtype = torch.float16
                    logger.info("Using float16 precision (bfloat16 not supported)")
            else:
                torch_dtype = torch.float32
                logger.info("Using float32 precision (non-CUDA device)")

            # Load model with specified dtype
            model_obj = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch_dtype,
                quantization_config=None,
                config=model_config,
                **provider_config.model_specific_config,
            )
            logger.info("Base model loaded")

            # Apply LoRA if configured
            if peft_config:
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model_obj.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):  # pylint: disable=unused-argument
                        output.requires_grad_(True)

                    model_obj.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

                logger.info("Applying LoRA configuration")
                model_obj = LoraModel(model_obj, peft_config, "default")
                logger.info("LoRA configuration applied")
            else:
                model_obj.gradient_checkpointing_enable()

            # Move model to device and return
            # model_obj.to(device=device)
            # logger.info(f"Model device: {next(model_obj.parameters()).device}")
            return model_obj
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}") from e

    def setup_training_args(
        self,
        config: TrainingConfig,
        provider_config: HuggingFacePostTrainingConfig,
        output_dir_path: Path | None,
        peft_config: LoraConfig | None,
        steps_per_epoch: int,
        model: AutoModelForCausalLM,
    ) -> SFTConfig:
        """Setup training arguments for distributed training.

        The FSDP (Fully Sharded Data Parallel) configuration is split into two parts:
        1. The fsdp_config dict which contains settings that are directly used by FSDP
        2. The fsdp string parameter which contains settings that are parsed by the trainer

        This split is necessary because the trainer only passes certain FSDP settings to the actual FSDP implementation.
        """
        logger.info("Configuring training arguments for distributed training")
        lr = 2e-5
        if config.optimizer_config:
            lr = config.optimizer_config.lr
            logger.info(f"Using custom learning rate: {lr}")

        # Validate data config
        if not config.data_config:
            raise ValueError("DataConfig is required for training")
        data_config = config.data_config

        # Calculate steps
        total_steps = steps_per_epoch * config.n_epochs
        max_steps = min(config.max_steps_per_epoch, total_steps)
        save_steps = max(1, steps_per_epoch // 5)  # Save 5 times per epoch
        logging_steps = max(1, steps_per_epoch // 50)  # Log 50 times per epoch

        logger.info("Training configuration:")
        logger.info(f"- Steps per epoch: {steps_per_epoch}")
        logger.info(f"- Total steps: {total_steps}")
        logger.info(f"- Max steps: {max_steps}")
        logger.info(f"- Save steps: {save_steps}")
        logger.info(f"- Logging steps: {logging_steps}")

        # Calculate optimal batch size based on available GPUs
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            effective_batch_size = max(1, data_config.batch_size // num_gpus)
            logger.info(f"Using {effective_batch_size} batch size per GPU (total batch size: {data_config.batch_size})")
        else:
            effective_batch_size = data_config.batch_size

        # Determine optimal precision settings
        if torch.cuda.is_available():
            fp16 = not torch.cuda.is_bf16_supported()
            bf16 = torch.cuda.is_bf16_supported()
            logger.info(f"Using {'bfloat16' if bf16 else 'float16'} precision")
        else:
            fp16 = False
            bf16 = False
            logger.info("Using float32 precision")

        # Configure save strategy
        save_strategy = "no"
        if output_dir_path:
            save_strategy = "steps"  # Save by steps for more frequent saves
            logger.info(f"Will save checkpoints to {output_dir_path}")

        # FSDP Configuration - Part 1: Direct FSDP settings
        # These settings are passed directly to the FSDP implementation
        fsdp_config = {
            # Enable CPU RAM efficient loading to reduce GPU memory usage during model loading
            "cpu_ram_efficient_loading": True,
            # Specify which transformer layer class to wrap with FSDP
            # This is crucial for proper sharding of the model
            "transformer_layer_cls_to_wrap": [model._no_split_modules[0]],
            # Use full sharding strategy for maximum memory efficiency
            "sharding_strategy": "FULL_SHARD",
            # Disable forward prefetch to reduce memory usage
            "forward_prefetch": False,
            # Limit all-gather operations to reduce memory spikes
            "limit_all_gathers": True,
            # Enable parameter offloading to CPU to reduce GPU memory usage
            "offload_param": True,
            # Ensure module states are synchronized across processes
            "sync_module_states": True,
            # Enable verbose logging for debugging
            "verbose": True,
            # State dict settings for better checkpoint handling
            "state_dict_type": "FULL_STATE_DICT",
            "state_dict_config": {
                "offload_to_cpu": True,  # Offload state dict to CPU during saving
                "rank0_only": True,  # Only rank 0 saves the state dict
            },
        }

        # Add LoRA-specific or full model FSDP settings
        if peft_config:
            # LoRA configuration - less aggressive sharding since LoRA is already memory efficient
            fsdp_config.update(
                {
                    "backward_prefetch": "backward_post",  # Prefetch after backward pass
                    "activation_checkpointing": False,  # No need for activation checkpointing with LoRA
                    "use_orig_params": False,  # Don't use original parameters for LoRA
                }
            )
        else:
            # Full model configuration - more aggressive memory optimization
            fsdp_config.update(
                {
                    "backward_prefetch": "backward_pre",  # Prefetch before backward pass
                    "activation_checkpointing": False,  # Use FSDP's built-in activation checkpointing
                    "use_orig_params": True,  # Use original parameters for full model
                }
            )

        # Set up training config
        training_config = SFTConfig(
            # FSDP Configuration - Part 2: Trainer-level FSDP settings
            # These settings are parsed by the trainer and passed to FSDP
            fsdp="full_shard auto_wrap offload",  # Enable full sharding, auto wrapping, and offloading
            # Pass the direct FSDP settings
            fsdp_config=fsdp_config,
            # Enable gradient checkpointing for memory efficiency
            gradient_checkpointing=provider_config.gradient_checkpointing,
            gradient_checkpointing_kwargs={
                "use_reentrant": False,  # Disable reentrant checkpointing for better memory efficiency
                "preserve_rng_state": False,  # Don't preserve RNG state to save memory
            }
            if provider_config.gradient_checkpointing
            else None,
            # Disable torch.compile as it can interfere with FSDP
            torch_compile=False,
            # Training parameters
            max_steps=max_steps,
            dataloader_num_workers=1,  # Single worker to avoid memory issues
            dataloader_pin_memory=False,  # Disable pin memory to reduce memory usage
            optim="adamw_torch",  # Use PyTorch's AdamW implementation
            output_dir=str(output_dir_path) if output_dir_path is not None else None,
            num_train_epochs=config.n_epochs,
            per_device_train_batch_size=effective_batch_size,
            fp16=fp16,
            bf16=bf16,
            eval_strategy="no",
            use_cpu=False,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=3,  # Keep last 3 checkpoints
            save_safetensors=True,
            save_only_model=True,  # Only save model state, not optimizer state for FSDP compatibility
            report_to="none",
            max_seq_length=provider_config.max_seq_length,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_steps=25,
            warmup_ratio=provider_config.warmup_ratio,
            weight_decay=provider_config.weight_decay,
            remove_unused_columns=False,
            dataset_text_field="text",
            load_best_model_at_end=False,
            metric_for_best_model="eval_loss",
            packing=False,
            greater_is_better=False,
            logging_steps=logging_steps,
            logging_first_step=True,
            logging_dir=str(output_dir_path / "logs") if output_dir_path else None,
            logging_nan_inf_filter=True,
            overwrite_output_dir=True,
        )

        return training_config

    def save_model(
        self,
        local_rank: int,
        model_path: str,
        model_obj: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizer,
        trainer: SFTTrainer,
        peft_config: LoraConfig | None,
        output_dir_path: Path,
    ) -> None:
        """Save the trained model with proper FSDP handling.

        This method handles saving both LoRA and full models with proper FSDP state dict handling.
        For LoRA models, it merges the weights with the base model before saving.

        Args:
            local_rank: Local rank of this process
            model_path: Path to the original model
            model_obj: The model to save
            tokenizer: Tokenizer to save
            trainer: The trainer instance
            peft_config: Optional LoRA configuration
            output_dir_path: Path to save the model

        Raises:
            RuntimeError: If model saving fails
        """
        logger.info("Saving final model")
        model_obj.config.use_cache = True
        save_path = output_dir_path / "final_model"
        logger.info(f"Saving model to {save_path}")

        # Ensure all processes are ready to save
        dist.barrier()

        try:
            if peft_config:
                logger.info("Merging LoRA weights with base model")
                # Get full state dict with FSDP handling
                sd_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FullyShardedDataParallel.state_dict_type(model_obj, StateDictType.FULL_STATE_DICT, sd_config):
                    state = model_obj.state_dict()

                if local_rank == 0:
                    try:
                        # Load a CPU copy of the base model for merging
                        logger.info("Loading CPU copy of base model for merging")
                        model_copy = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            device_map="cpu",  # Ensure CPU loading
                            torch_dtype=torch.float32,  # Use float32 for better precision during merging
                        )
                        model_copy = LoraModel(model_copy, peft_config, "default")

                        # Load the trained state and merge
                        logger.info("Loading trained state and merging weights")
                        model_copy.load_state_dict(state)
                        merged_model = model_copy.merge_and_unload(progressbar=True)

                        # Save the merged model and tokenizer
                        logger.info("Saving merged model and tokenizer")
                        merged_model.save_pretrained(save_path, safe_serialization=True)
                        tokenizer.save_pretrained(save_path)

                        # Clean up
                        del model_copy
                        logger.info("Successfully saved merged LoRA model and tokenizer")
                    except Exception as e:
                        logger.error(f"Failed to save merged LoRA model: {str(e)}")
                        raise
            else:
                logger.info("Saving full model with FSDP")
                # For full model, use FSDP's state dict handling
                if local_rank == 0:
                    try:
                        model_obj.save_pretrained(save_path, safe_serialization=True)
                        tokenizer.save_pretrained(save_path)
                        logger.info("Successfully saved full model and tokenizer")
                    except Exception as e:
                        logger.error(f"Failed to save full model: {str(e)}")
                        raise
        finally:
            # Ensure all processes wait for saving to complete
            dist.barrier()
            logger.info("Model saving completed")

    async def train(
        self,
        model: str,
        output_dir: str | None,
        job_uuid: str,
        lora_config: LoraFinetuningConfig,
        config: TrainingConfig,
        provider_config: HuggingFacePostTrainingConfig,
    ) -> tuple[dict[str, Any], list[Checkpoint] | None]:
        """Train a model using HuggingFace's SFTTrainer with distributed training.

        The distributed training setup works as follows:
        1. Parse the device list to determine number of GPUs
        2. Use torch.multiprocessing.spawn to launch one process per GPU
        3. Each process runs _run_training_sync with a unique rank
        4. The processes coordinate through NCCL backend
        5. FSDP handles model sharding across GPUs
        6. Only rank 0 handles saving checkpoints and logging

        Args:
            model: The model identifier to load
            output_dir: Optional directory to save checkpoints
            job_uuid: Unique identifier for this training job
            lora_config: LoRA configuration for parameter-efficient fine-tuning
            config: General training configuration
            provider_config: Provider-specific configuration
        Returns:
            tuple: (memory_stats, checkpoints)
        """

        if provider_config.distributed_backend != "fsdp":
            raise RuntimeError("Must enable FSDP as distributed backend to use this recipe")

        # Configure NCCL logging based on debug settings
        configure_nccl_logging(self.enable_nccl_debug, self.nccl_debug_subsys)

        # Parse device list to determine number of GPUs
        devices = [d.strip() for d in provider_config.device.split(",")]
        world_size = len(devices)
        logger.info(f"Using {world_size} devices: {devices}")

        output_dir_path = None
        if output_dir:
            output_dir_path = Path(output_dir)

        # Track memory stats on first GPU
        memory_stats = {
            "initial": get_memory_stats(torch.device("cuda:0")),
            "after_training": None,
            "final": None,
        }

        # Configure LoRA if specified
        peft_config = None
        if lora_config:
            peft_config = LoraConfig(
                lora_alpha=lora_config.alpha,
                lora_dropout=0.1,
                r=lora_config.rank,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_config.lora_attn_modules,
            )

        # Validate data config
        if not config.data_config:
            raise ValueError("DataConfig is required for training")

        try:
            # Launch distributed training processes
            # torch.multiprocessing.spawn will:
            # 1. Create world_size number of processes
            # 2. Call _run_training_sync for each process
            # 3. Pass unique local_rank to each process
            # 4. Handle process coordination and cleanup
            logger.info("Starting distributed training processes")
            torch.multiprocessing.spawn(
                self._run_training_sync,
                args=(
                    world_size,
                    model,
                    provider_config.model_dump(),
                    peft_config,
                    config.model_dump(),
                    output_dir_path,
                ),
                nprocs=world_size,
                join=True,  # Wait for all processes to complete
            )

            memory_stats["after_training"] = get_memory_stats(torch.device("cuda:0"))

            # Create checkpoint on rank 0
            checkpoints = None
            if output_dir_path:
                checkpoint = Checkpoint(
                    identifier=f"{model}-sft-{config.n_epochs}",
                    created_at=datetime.now(timezone.utc),
                    epoch=config.n_epochs,
                    post_training_job_id=job_uuid,
                    path=str(output_dir_path / "merged_model"),
                )
                checkpoints = [checkpoint]

            return memory_stats, checkpoints
        finally:
            memory_stats["final"] = get_memory_stats(torch.device("cuda:0"))
            gc.collect()
