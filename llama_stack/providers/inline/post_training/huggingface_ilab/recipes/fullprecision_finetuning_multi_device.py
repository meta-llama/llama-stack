import tempfile
import typing
from pathlib import Path
from typing import Callable

import transformers
from transformers.configuration_utils import PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from llama_stack.apis.common.job_types import JobStatus
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.post_training.post_training import AlgorithmConfig, TrainingConfig

STORAGE_SUBDIRS = ["checkpoints", "data", "logs", "hf_cache"]
VALIDATED_MODEL_ARCHS = ["LlamaForCausalLM", "GraniteForCausalLM"]

SomePretrainedTokenizer: typing.TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast


class FullPrecisionFineTuning:
    # TODO: set HF storage in HF utilities to this object's storage so that we can clean it up automatically
    # TODO: set up logging utils, replace print statements with logging with the right level
    def __init__(
        self,
        model: str,
        training_config: TrainingConfig,
        logger_config: dict[str, typing.Any],
        storage_dir: Path | None,
        algorithm_config: AlgorithmConfig,  # type: ignore
        datasets_api: Datasets,
        datasetsio_api: DatasetIO,
    ):
        self.model_name_or_path = model
        self.training_config = training_config
        self.logger_config = logger_config
        if storage_dir:
            self.storage_dir = storage_dir
        else:
            self.storage_dir = Path(tempfile.mkdtemp())
        self.__setup_storage()

        self.datasets_api = datasets_api
        self.datasetio_api = datasetsio_api
        self.loaded_dataset: typing.Any = None  # should be a list of dicts but shape can be weird

    def __setup_storage(self):
        for subdir in STORAGE_SUBDIRS:
            new_subdir = self.storage_dir / subdir
            new_subdir.mkdir(exist_ok=True, parents=True)

    @property
    def checkpoint_dir(self):
        return self.storage_dir / "checkpoints"

    @property
    def data_dir(self):
        return self.storage_dir / "data"

    @property
    def logs_dir(self):
        return self.storage_dir / "logs"

    @staticmethod
    def check_model_arch_validated(model_config: PretrainedConfig) -> bool:
        if model_config.architectures is None:
            return False

        for arch in model_config.architectures:
            if arch in VALIDATED_MODEL_ARCHS:
                return True

        return False

    def __try_load_config(self) -> PretrainedConfig:
        try:
            model_config: PretrainedConfig = transformers.AutoConfig.from_pretrained(self.model_name_or_path)
        except OSError:
            print(
                f"Attempted to load model config for ({self.model_name_or_path}) but failed. Model config will be loaded by `AutoConfig.from_pretrained()`"
            )
            raise

        return model_config

    def __try_load_tokenizer(self) -> SomePretrainedTokenizer:
        try:
            tokenizer: SomePretrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_name_or_path, use_fast=True
            )
        except OSError:
            print(
                f"Attempted to load model tokenizer for ({self.model_name_or_path}) but failed. Model tokenizer will be loaded by `AutoTokenizer.from_pretrained()`"
            )
            raise

        return tokenizer

    @staticmethod
    def check_tokenizer_has_chat_template(tokenizer: SomePretrainedTokenizer) -> bool:
        if not hasattr(tokenizer, "chat_template"):
            return False

        if tokenizer.chat_template is None:
            return False

        return True

    async def load_dataset_from_datasetsio(self):
        dataset = await self.datasetio_api.get_rows_paginated(
            dataset_id=self.training_config.data_config.dataset_id, rows_in_page=-1
        )
        self.loaded_dataset = dataset.rows

    def preflight(self, set_status_callback: Callable[[JobStatus], None]):
        """
        A synchronous "preflight" operation from the recipe runs and does the following checks:
        1. (future) validates that the host has access to sufficient hardware. For now, assume that an administrator has "cleared" the deployment for any requests it could get. In the future, check:
            1. Cards exist
            2. Cards have enough memory
            3. Cards are idle
            4. Cards have silicon for functions (bfloat16 tensor cores, support FA)
            5. Cards are functional
        2. Validates that model is available from HF.
        3. Validates that model is a verified architecture (warns user if not).
        4. Validates that model's tokenizer exists.
        5. Validates that the model's tokenizer has a chat template.
        6. Validates that the model's chat template can render a sample from the dataset.
        """

        model_config = self.__try_load_config()
        if not self.check_model_arch_validated(model_config=model_config):
            # could raise Error if we need a strong check against this.
            print(
                f"Input model ({self.model_name_or_path}) architecture ({model_config.architectures}) is not among validated architectures."
            )

        model_tokenizer = self.__try_load_tokenizer()
        if not self.check_tokenizer_has_chat_template(model_tokenizer):
            raise RuntimeError(
                f"Input model ({self.model_name_or_path})'s tokenizer ({model_tokenizer.__name__}) has no chat template from associated `tokenizer_config.json`"
            )

        try:
            rendered_sample = model_tokenizer.apply_chat_template(self.loaded_dataset[0]["messages"])
        except Exception:
            # catching / raising bare exception because 'apply_chat_template' can raise ValueError or TypeError; want to report the same thing regardless.
            print(
                f"Input model ({self.model_name_or_path})'s tokenizer ({model_tokenizer.__name__}) could not tokenize dataset sample. Please make sure that sample is OpenAI 'chat' formatted."
            )
            raise

        # Success! Preflight checks haven't caught any immediate problems.
        set_status_callback(JobStatus.scheduled)

    def setup(self):
        """
        A synchronous data preprocessing operation that runs in a kernel-scheduled background thread and does the following:
        1. Requests all rows of data from datasetsio API
        2. Ports data into a `huggingface.datasets` object
        3. Instantiates the model tokenizer
        4. `dataset.map`'s the input data into chat template format
        5. generates labels, masks for each sample
        6. writes dataset to temporary storage
        """
        pass

    async def train(self, set_status_callback: Callable[[JobStatus], None]):
        """
        An asynchronous instance method that creates and watches a `torchrun` subprocess that's training the input model.
        """
        set_status_callback(JobStatus.completed)
