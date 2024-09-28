import logging
from typing import Any

from llama_stack.apis.inference.inference import CompletionResponse, CompletionResponseStreamChunk, LogProbConfig, ChatCompletionResponse, ChatCompletionResponseStreamChunk, EmbeddingsResponse
from llama_stack.apis.inference import Inference

from .config import VLLMConfig

from llama_models.llama3.api.datatypes import InterleavedTextMedia, Message, ToolChoice, ToolDefinition, ToolPromptFormat


log = logging.getLogger(__name__)


class VLLMInferenceImpl(Inference):
    """Inference implementation for vLLM."""
    def __init__(self, config: VLLMConfig):
        self.config = config

    async def initialize(self):
        log.info("Initializing vLLM inference adapter")
        pass

    async def completion(self, model: str, content: InterleavedTextMedia, sampling_params: Any | None = ..., stream: bool | None = False, logprobs: LogProbConfig | None = None) -> CompletionResponse | CompletionResponseStreamChunk:
        log.info("vLLM completion")
        return None

    async def chat_completion(self, model: str, messages: list[Message], sampling_params: Any | None = ..., tools: list[ToolDefinition] | None = ..., tool_choice: ToolChoice | None = ..., tool_prompt_format: ToolPromptFormat | None = ..., stream: bool | None = False, logprobs: LogProbConfig | None = None) -> ChatCompletionResponse | ChatCompletionResponseStreamChunk:
        log.info("vLLM chat completion")
        return None

    async def embeddings(self, model: str, contents: list[InterleavedTextMedia]) -> EmbeddingsResponse:
        log.info("vLLM embeddings")
