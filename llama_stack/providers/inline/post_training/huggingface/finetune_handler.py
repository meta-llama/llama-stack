#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import asyncio
import json
import os
from typing import Any

from llama_stack.apis.post_training import TrainingConfig
from llama_stack.providers.inline.post_training.huggingface.config import HuggingFacePostTrainingConfig
from llama_stack.providers.inline.post_training.huggingface.recipes.finetune_multi_device import (
    HFFinetuningMultiDevice,
)
from llama_stack.providers.utils.scheduler import JobStatus


async def train(
    job_uuid,
    model,
    checkpoint_dir,
    training_config,
    provider_config,
    algorithm_config,
    data,
    enable_nccl_debug=False,
    nccl_debug_subsys="NONE",
):
    """Handler function for HuggingFace training that can be called by torchrun.

    This is extracted from the supervised_fine_tune method in the HuggingFacePostTrainingImpl class.
    It follows the same flow, but is designed to be called directly from a script.

    Args:
        job_uuid: Unique ID for this job
        model: Model to train
        checkpoint_dir: Directory to save checkpoints to
        training_config: Training configuration
        provider_config: Provider configuration
        algorithm_config: Algorithm configuration
        data: the dataset rows to be processed
        enable_nccl_debug: Whether to enable NCCL debugging
        nccl_debug_subsys: NCCL subsystem to debug
    """
    # Get rank information when running distributed
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    parsed_data: list[dict[str, Any]] = json.loads(data)

    # Set up callback functions with rank information
    def on_log_message_cb(msg):
        print(f"[RANK {local_rank}] {msg}", flush=True)

    def on_status_change_cb(status):
        print(f"[RANK {local_rank}] Status: {status}", flush=True)

    def on_artifact_collected_cb(artifact):
        print(f"[RANK {local_rank}] Artifact: {artifact}", flush=True)

    on_log_message_cb("Starting HF finetuning")

    recipe_obj = HFFinetuningMultiDevice(
        job_uuid=job_uuid, enable_nccl_debug=enable_nccl_debug, nccl_debug_subsys=nccl_debug_subsys, data=parsed_data
    )

    resources_allocated, checkpoints = await recipe_obj.train(
        model=model,
        output_dir=checkpoint_dir,
        job_uuid=job_uuid,
        lora_config=algorithm_config,
        config=training_config,
        provider_config=provider_config,
    )

    def resources_stats_to_artifact(resources_stats):
        return {
            "type": "resources_stats",
            "name": "resources_stats",
            "metadata": resources_stats,
        }

    def checkpoint_to_artifact(checkpoint):
        return {
            "type": "checkpoint",
            "name": checkpoint.identifier,
            "uri": checkpoint.path,
            "metadata": dict(checkpoint),
        }

    on_artifact_collected_cb(resources_stats_to_artifact(resources_allocated))
    if checkpoints:
        for checkpoint in checkpoints:
            artifact = checkpoint_to_artifact(checkpoint)
            on_artifact_collected_cb(artifact)

    on_status_change_cb(JobStatus.completed)
    on_log_message_cb("HF finetuning completed")


async def main():
    parser = argparse.ArgumentParser(description="Run HuggingFace training with torchrun.")
    parser.add_argument("--job_uuid", type=str, required=True, help="Job UUID")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory to save checkpoints")
    parser.add_argument("--training_config", type=str, required=True, help="Training config JSON")
    parser.add_argument("--provider_config", type=str, required=True, help="Provider config JSON")
    parser.add_argument("--algorithm_config", type=str, help="Algorithm config JSON")
    parser.add_argument("--enable_nccl_debug", action="store_true", help="Enable NCCL debugging")
    parser.add_argument("--nccl_debug_subsys", type=str, default="NONE", help="NCCL subsystem to debug")
    parser.add_argument("--data", type=str, required=True)

    args = parser.parse_args()

    # Parse JSON configs
    try:
        training_config = TrainingConfig.model_validate_json(args.training_config)
    except Exception as e:
        print(f"Error parsing training_config: {e}")
        print(f"Received: {args.training_config}")
        raise

    try:
        provider_config = HuggingFacePostTrainingConfig.model_validate_json(args.provider_config)
    except Exception as e:
        print(f"Error parsing provider_config: {e}")
        print(f"Received: {args.provider_config}")
        raise

    algorithm_config = None
    if args.algorithm_config:
        try:
            algorithm_config = json.loads(args.algorithm_config)
        except json.JSONDecodeError as e:
            print(f"Error parsing algorithm_config: {e}")
            print(f"Received: {args.algorithm_config}")
            raise

    # In a real implementation, you would get these from somewhere
    # For now, we'll pass None and handle it in the train function
    datasetio_api = None
    datasets_api = None

    # Print arguments for debugging
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:  # Only the main process prints
        print("Starting training with arguments:")
        print(f"  job_uuid: {args.job_uuid}")
        print(f"  model: {args.model}")
        print(f"  checkpoint_dir: {args.checkpoint_dir}")
        print(f"  enable_nccl_debug: {args.enable_nccl_debug}")
        print(f"  nccl_debug_subsys: {args.nccl_debug_subsys}")

    await train(
        job_uuid=args.job_uuid,
        model=args.model,
        checkpoint_dir=args.checkpoint_dir,
        training_config=training_config,
        provider_config=provider_config,
        algorithm_config=algorithm_config,
        datasetio_api=datasetio_api,
        datasets_api=datasets_api,
        enable_nccl_debug=args.enable_nccl_debug,
        nccl_debug_subsys=args.nccl_debug_subsys,
        data=args.data,
    )


if __name__ == "__main__":
    asyncio.run(main())
