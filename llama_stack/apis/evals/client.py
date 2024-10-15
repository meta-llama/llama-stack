# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json

import fire
import httpx
from termcolor import cprint

from .evals import *  # noqa: F403
import base64
import mimetypes
import os

from ..datasets.client import DatasetsClient


def data_url_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as file:
        file_content = file.read()

    base64_content = base64.b64encode(file_content).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)

    data_url = f"data:{mime_type};base64,{base64_content}"

    return data_url


class EvaluationClient(Evals):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def run_evals(
        self,
        eval_task_config: EvaluateTaskConfig,
    ) -> EvaluateResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/evals/run_eval_task",
                json={
                    "eval_task_config": json.loads(eval_task_config.json()),
                },
                headers={"Content-Type": "application/json"},
                timeout=3600,
            )
            response.raise_for_status()
            return EvaluateResponse(**response.json())

    async def run_scorer(
        self,
        dataset_config: EvaluateDatasetConfig,
        eval_scoring_config: EvaluateScoringConfig,
    ) -> EvaluateResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/evals/run_scorer",
                json={
                    "dataset_config": json.loads(dataset_config.json()),
                    "eval_scoring_config": json.loads(eval_scoring_config.json()),
                },
                headers={"Content-Type": "application/json"},
                timeout=3600,
            )
            response.raise_for_status()
            return EvaluateResponse(**response.json())


async def run_main(host: str, port: int, eval_dataset_path: str = ""):
    client = EvaluationClient(f"http://{host}:{port}")
    dataset_client = DatasetsClient(f"http://{host}:{port}")

    # Full Eval Task
    # 1. register custom dataset
    response = await dataset_client.create_dataset(
        dataset_def=CustomDatasetDef(
            identifier="mmlu-simple-eval-en",
            url="https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv",
        ),
    )
    cprint(f"datasets/create: {response}", "cyan")

    # # 2. run evals on the registered dataset
    eval_task_config = EvaluateTaskConfig(
        dataset_config=EvaluateDatasetConfig(
            dataset_identifier="mmlu-simple-eval-en",
            row_limit=3,
        ),
        processor_config=EvaluateProcessorConfig(
            processor_identifier="mmlu",
        ),
        generation_config=EvaluateModelGenerationConfig(
            model="Llama3.1-8B-Instruct",
        ),
        scoring_config=EvaluateScoringConfig(
            scorer_config_list=[
                EvaluateSingleScorerConfig(scorer_name="accuracy"),
                EvaluateSingleScorerConfig(scorer_name="random"),
            ]
        ),
    )
    response = await client.run_evals(
        eval_task_config=eval_task_config,
    )
    for k, v in response.eval_result.metrics.items():
        cprint(f"{k}: {v}", "green")

    # Scoring Task
    # 1. register huggingface dataset
    response = await dataset_client.create_dataset(
        dataset_def=HuggingfaceDatasetDef(
            identifier="Llama-3.1-8B-Instruct-evals__mmlu_pro__details",
            dataset_path="meta-llama/Llama-3.1-8B-Instruct-evals",
            dataset_name="Llama-3.1-8B-Instruct-evals__mmlu_pro__details",
            rename_columns_map={
                "output_parsed_answer": "generated_answer",
                "input_correct_responses": "expected_answer",
            },
            kwargs={"split": "latest"},
        )
    )
    cprint(response, "cyan")

    # register custom dataset from file path
    response = await dataset_client.create_dataset(
        dataset_def=CustomDatasetDef(
            identifier="rag-evals",
            url=data_url_from_file(eval_dataset_path),
        )
    )
    cprint(response, "cyan")

    # 2. run evals on the registered dataset
    response = await client.run_scorer(
        dataset_config=EvaluateDatasetConfig(
            dataset_identifier="rag-evals",
            row_limit=10,
        ),
        eval_scoring_config=EvaluateScoringConfig(
            scorer_config_list=[
                EvaluateSingleScorerConfig(scorer_name="accuracy"),
                EvaluateSingleScorerConfig(
                    scorer_name="braintrust::answer-correctness"
                ),
            ]
        ),
    )

    for k, v in response.eval_result.metrics.items():
        cprint(f"{k}: {v}", "green")


def main(host: str, port: int, eval_dataset_path: str = ""):
    asyncio.run(run_main(host, port, eval_dataset_path))


if __name__ == "__main__":
    fire.Fire(main)
