from dataclasses import dataclass
from functools import partial
from typing import Generator, List, Optional

from models.llama3_1.api.chat_format import ChatFormat
from models.llama3_1.api.datatypes import Message
from models.llama3_1.api.tokenizer import Tokenizer

from .api.config import GeneratorArgs
from .generation import Llama
from .parallel_utils import ModelParallelProcessGroup


@dataclass
class InferenceArgs:
    messages: List[Message]
    temperature: float
    top_p: float
    max_gen_len: int
    logprobs: bool


class ModelRunner:
    def __init__(self, llama):
        self.llama = llama

    # the `task` object is the same that is sent to `ModelParallelProcessGroup.run_inference()`
    def __call__(self, task: InferenceArgs):
        return self.llama.chat_completion(
            task.messages,
            task.temperature,
            task.top_p,
            task.max_gen_len,
            task.logprobs,
        )


def init_model_cb(args: GeneratorArgs):
    llama = Llama.build(
        args.ckpt_dir,
        args.tokenizer_path,
        args.max_seq_len,
        args.max_batch_size,
    )
    return ModelRunner(llama)


class LlamaModelParallelGenerator:
    """
    This abstraction exists so
     - we can run model parallel code without needing to run the CLIs via torchrun
     - this also enables use model parallel code within a notebook context.

    A Context Manager is used to ensure that the model parallel process is started and stopped
    correctly. This does make the ergonomics a little awkward, because it isn't immediately
    clear at the callsite why we need to use a context manager.
    """

    def __init__(self, args: GeneratorArgs):
        self.args = args

        # this is a hack because Agent's loop uses this to tokenize and check if input is too long
        # while the tool-use loop is going
        self.formatter = ChatFormat(Tokenizer(self.args.tokenizer_path))

    def start(self):
        self.__enter__()

    def stop(self):
        self.__exit__(None, None, None)

    def __enter__(self):
        self.group = ModelParallelProcessGroup(
            self.args.model_parallel_size,
            init_model_cb=partial(init_model_cb, self.args),
        )
        self.group.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.group.stop()

    def chat_completion(
        self,
        messages: List[Message],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> Generator:
        req_obj = InferenceArgs(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
            logprobs=logprobs,
        )

        gen = self.group.run_inference(req_obj)
        yield from gen
