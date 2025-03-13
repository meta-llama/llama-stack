import asyncio
import tempfile
import typing
from asyncio import subprocess
from pathlib import Path
from typing import Callable

import datasets
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
TMP_DATA_FILE_NAME = "data.jsonl"

SomePretrainedTokenizer: typing.TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast

# TODO: set HF storage in HF utilities to this object's storage so that we can clean it up automatically
# TODO: set up logging utils, replace print statements with logging with the right level


class FullPrecisionFineTuning:
    """Implement full-precision (bfloat16) training.

    Uses subprocessing to launch `torchrun` processes for model tuning, SPMD.
    """

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
        """Check whether input model architecture from config is among the pre-validated architectures.

        Args:
            model_config (PretrainedConfig): input model config object

        Returns:
            bool: whether the model architecture is known to work with this training implementation.
        """
        if model_config.architectures is None:
            return False

        for arch in model_config.architectures:
            if arch in VALIDATED_MODEL_ARCHS:
                return True

        return False

    def __try_load_config(self) -> PretrainedConfig:
        """Attempt to load model config via model's name or path.

        Returns:
            PretrainedConfig: model config associated with model.
        """
        try:
            model_config: PretrainedConfig = transformers.AutoConfig.from_pretrained(self.model_name_or_path)
        except OSError:
            print(
                f"Attempted to load model config for ({self.model_name_or_path}) but failed. Model config will be loaded by `AutoConfig.from_pretrained()`"
            )
            raise

        return model_config

    def __try_load_tokenizer(self) -> SomePretrainedTokenizer:
        """Attempt to load tokenizer via model's name or path.

        Returns:
            SomePretrainedTokenizer: tokenizer associated with input model name.
        """
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
        """Checks for existence of chat template on tokenizer object.

        Args:
            tokenizer (SomePretrainedTokenizer): Model tokenizer

        Returns:
            bool: Whether 'chat_template' instance member exists and is not None.
        """
        if not hasattr(tokenizer, "chat_template"):
            return False

        if tokenizer.chat_template is None:
            return False

        return True

    async def load_dataset_from_datasetsio(self):
        """Loads all dataset rows from datasetio API. Sets 'loaded_dataset' in object."""

        dataset = await self.datasetio_api.get_rows_paginated(
            dataset_id=self.training_config.data_config.dataset_id, rows_in_page=-1
        )
        self.loaded_dataset = dataset.rows

    def preflight(self, set_status_callback: Callable[[JobStatus], None]):
        """Set of checks that should run before any heavier-weight preprocessing runs to validate starting state.

        Checks the following:
            1. Model config is available from Huggingface by the model's name.
            2. Model's architecture (from config) is among "validated" architectures.
            3. Model's tokenizer can be downloaded from Huggingface by the model's name.
            4. Tokenizer has a 'chat_template' available (we don't currently support BYO chat template).
            5. A single data sample can successfully be rendered without raising an error.

        In the future, it's be great for this method to also do some system-checking further up the call stack, like:
            1. Cards exist
            2. Cards have enough memory
            3. Cards are idle
            4. Cards have silicon for functions (bfloat16 tensor cores, support FA)
            5. Cards are functional

        Args:
            set_status_callback (Callable[[JobStatus], None]): Sets job status in calling 'Impl' class' ref to this job.

        Raises:
            RuntimeError: If tokenizer doesn't have chat template available.
            OSError: Can be raised via this function if config or tokenizer not available via model's name.
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
            _ = model_tokenizer.apply_chat_template(self.loaded_dataset[0]["messages"])
        except Exception:
            # catching / raising bare exception because 'apply_chat_template' can raise ValueError or TypeError; want to report the same thing regardless.
            print(
                f"Input model ({self.model_name_or_path})'s tokenizer ({model_tokenizer.__name__}) could not tokenize dataset sample. Please make sure that sample is OpenAI 'chat' formatted."
            )
            raise

        # Success! Preflight checks haven't caught any immediate problems.
        set_status_callback(JobStatus.scheduled)

    @staticmethod
    def __tokenize_and_generate_labels_and_mask(
        tokenizer: SomePretrainedTokenizer,
        sample: list[dict[typing.Any, typing.Any]],  # TODO: type dict correctly.
    ):
        """Helper method for preparing a single chat sample for model training.

        Assumed (but not required) to have been called from a `dataset.map()` call.
        Tokenizes sample using `tokenizer.apply_chat_template()` and uses that output
        for the associated labels.

        Creates 'attention_mask' and 'loss_mask' of ones (doesn't mask out non-assistant messages).

        Args:
            tokenizer (SomePretrainedTokenizer): Tokenizer associated with model
            sample (list[dict[typing.Any, typing.Any]]): Input OpenAI chat conversation dataset sample.

        Returns:
            dict[str, list[int]]: Of shape {input_ids, labels, attention_mask, loss_mask}
        """

        input_ids = tokenizer.apply_chat_template(conversation=sample, tokenize=True)
        input_ids = typing.cast(
            list[int], input_ids
        )  # I know what the output will be, and this makes the typing system happy.
        labels = input_ids[:]
        attention_mask = [1] * len(labels)  # == [1 for _ in range(len(labels))]
        loss_mask = [1] * len(labels)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask, "loss_mask": loss_mask}

    def __setup_data(self):
        """Helper method for loading and tokenizing dataset from .get_rows_paginated() API.

        Tokenizes data using special tokens and chat template from model tokenizer.
        Doesn't do specialized sample preprocessing re: chat message types for loss calculation.

        Returns:
            datasets.Dataset: Dataset object w/ data prepared for training.
        """

        dataset = datasets.Dataset.from_list(self.loaded_dataset)
        model_tok = self.__try_load_tokenizer()

        # NOTE: not implementing as batched for the moment; need to know how batching impacts memory usage on machine.
        dataset = dataset.map(lambda x: self.__tokenize_and_generate_labels_and_mask(tokenizer=model_tok, sample=x))
        return dataset

    def setup(self):
        """Data preprocessing to prepare for model training. Writes data to local cache dir to be read by SPMD training processes later."""

        dataset = self.__setup_data()
        dataset.to_json(path_or_buf=self.data_dir / TMP_DATA_FILE_NAME)

    async def train(
        self,
        set_status_callback: Callable[[JobStatus], None],
        set_subproc_ref_callback: Callable[[subprocess.Process], None],
    ):
        """Subprocesses `torchrun` as async function and updates state of calling `Impl` class.

        Args:
            set_status_callback (Callable[[JobStatus], None]): Sets job status in calling 'Impl' class' ref to this job.
            set_subproc_ref_callback (Callable[[subprocess.Process], None]): Sets subprocess reference in 'Impl' class' ref to this job
        """

        training_subproc = await asyncio.create_subprocess_exec(
            "echo 'yay Im running in a subprocess: $$'; sleep 30; echo 'exiting process $$'"
        )
        set_subproc_ref_callback(training_subproc)
        await training_subproc.wait()
        set_status_callback(JobStatus.completed)
