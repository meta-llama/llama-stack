import httpx
import uuid

from typing import AsyncGenerator

from ollama import AsyncClient

from llama_models.llama3_1.api.datatypes import (
    BuiltinTool,
    CompletionMessage, 
    Message, 
    StopReason,
    ToolCall,
)
from llama_models.llama3_1.api.tool_utils import ToolUtils

from .api.config import OllamaImplConfig
from .api.endpoints import (
    ChatCompletionResponse,
    ChatCompletionRequest,
    ChatCompletionResponseStreamChunk,
    CompletionRequest,
    Inference,
)



class OllamaInference(Inference):

    def __init__(self, config: OllamaImplConfig) -> None:
        self.config = config
        self.model = config.model

    async def initialize(self) -> None:
        self.client = AsyncClient(host=self.config.url)
        try:
            status = await self.client.pull(self.model)
            assert status['status'] == 'success', f"Failed to pull model {self.model} in ollama"
        except httpx.ConnectError:
            print("Ollama Server is not running, start it using `ollama serve` in a separate terminal")
            raise

    async def shutdown(self) -> None:
        pass

    async def completion(self, request: CompletionRequest) -> AsyncGenerator:
        raise NotImplementedError()

    def _messages_to_ollama_messages(self, messages: list[Message]) -> list:
        ollama_messages = []
        for message in messages:
            ollama_messages.append(
                {"role": message.role, "content": message.content}
            )

        return ollama_messages
    
    async def chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator:
        if not request.stream:
            r = await self.client.chat(
                model=self.model,
                messages=self._messages_to_ollama_messages(request.messages),
                stream=False
            )
            completion_message = decode_assistant_message_from_content(
                r['message']['content']
            )
            
            yield ChatCompletionResponse(
                completion_message=completion_message,
                logprobs=None,
            )
        else:
            raise NotImplementedError()


#TODO: Consolidate this with impl in llama-models
def decode_assistant_message_from_content(content: str) -> CompletionMessage:
    ipython = content.startswith("<|python_tag|>")
    if ipython:
        content = content[len("<|python_tag|>") :]

    if content.endswith("<|eot_id|>"):
        content = content[: -len("<|eot_id|>")]
        stop_reason = StopReason.end_of_turn
    elif content.endswith("<|eom_id|>"):
        content = content[: -len("<|eom_id|>")]
        stop_reason = StopReason.end_of_message
    else:
        # Ollama does not return <|eot_id|>
        # and hence we explicitly set it as the default.
        #TODO: Check for StopReason.out_of_tokens
        stop_reason = StopReason.end_of_turn

    tool_name = None
    tool_arguments = {}

    custom_tool_info = ToolUtils.maybe_extract_custom_tool_call(content)
    if custom_tool_info is not None:
        tool_name, tool_arguments = custom_tool_info
        # Sometimes when agent has custom tools alongside builin tools
        # Agent responds for builtin tool calls in the format of the custom tools
        # This code tries to handle that case
        if tool_name in BuiltinTool.__members__:
            tool_name = BuiltinTool[tool_name]
            tool_arguments = {
                "query": list(tool_arguments.values())[0],
            }
    else:
        builtin_tool_info = ToolUtils.maybe_extract_builtin_tool_call(content)
        if builtin_tool_info is not None:
            tool_name, query = builtin_tool_info
            tool_arguments = {
                "query": query,
            }
            if tool_name in BuiltinTool.__members__:
                tool_name = BuiltinTool[tool_name]
        elif ipython:
            tool_name = BuiltinTool.code_interpreter
            tool_arguments = {
                "code": content,
            }

    tool_calls = []
    if tool_name is not None and tool_arguments is not None:
        call_id = str(uuid.uuid4())
        tool_calls.append(
            ToolCall(
                call_id=call_id,
                tool_name=tool_name,
                arguments=tool_arguments,
            )
        )
        content = ""

    if stop_reason is None:
        stop_reason = StopReason.out_of_tokens

    return CompletionMessage(
        content=content,
        stop_reason=stop_reason,
        tool_calls=tool_calls,
    )
