# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from enum import Enum
from typing import (
    Annotated,
    Any,
    Literal,
    Protocol,
    runtime_checkable,
)

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict

from llama_stack.apis.common.content_types import ContentDelta, InterleavedContent, InterleavedContentItem
from llama_stack.apis.common.responses import Order
from llama_stack.apis.models import Model
from llama_stack.apis.telemetry import MetricResponseMixin
from llama_stack.models.llama.datatypes import (
    BuiltinTool,
    StopReason,
    ToolCall,
    ToolDefinition,
    ToolParamDefinition,
    ToolPromptFormat,
)
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol
from llama_stack.schema_utils import json_schema_type, register_schema, webmethod

register_schema(ToolCall)
register_schema(ToolParamDefinition)
register_schema(ToolDefinition)

from enum import StrEnum


@json_schema_type
class GreedySamplingStrategy(BaseModel):
    type: Literal["greedy"] = "greedy"


@json_schema_type
class TopPSamplingStrategy(BaseModel):
    type: Literal["top_p"] = "top_p"
    temperature: float | None = Field(..., gt=0.0)
    top_p: float | None = 0.95


@json_schema_type
class TopKSamplingStrategy(BaseModel):
    type: Literal["top_k"] = "top_k"
    top_k: int = Field(..., ge=1)


SamplingStrategy = Annotated[
    GreedySamplingStrategy | TopPSamplingStrategy | TopKSamplingStrategy,
    Field(discriminator="type"),
]
register_schema(SamplingStrategy, name="SamplingStrategy")


@json_schema_type
class SamplingParams(BaseModel):
    """Sampling parameters.

    :param strategy: The sampling strategy.
    :param max_tokens: The maximum number of tokens that can be generated in the completion. The token count of
        your prompt plus max_tokens cannot exceed the model's context length.
    :param repetition_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens
        based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    :param stop: Up to 4 sequences where the API will stop generating further tokens.
        The returned text will not contain the stop sequence.
    """

    strategy: SamplingStrategy = Field(default_factory=GreedySamplingStrategy)

    max_tokens: int | None = 0
    repetition_penalty: float | None = 1.0
    stop: list[str] | None = None


class LogProbConfig(BaseModel):
    """

    :param top_k: How many tokens (for each position) to return log probabilities for.
    """

    top_k: int | None = 0


class QuantizationType(Enum):
    """Type of model quantization to run inference with.

    :cvar bf16: BFloat16 typically this means _no_ quantization
    :cvar fp8_mixed: 8-bit floating point quantization with mixed precision
    :cvar int4_mixed: 4-bit integer quantization with mixed precision
    """

    bf16 = "bf16"
    fp8_mixed = "fp8_mixed"
    int4_mixed = "int4_mixed"


@json_schema_type
class Fp8QuantizationConfig(BaseModel):
    type: Literal["fp8_mixed"] = "fp8_mixed"


@json_schema_type
class Bf16QuantizationConfig(BaseModel):
    type: Literal["bf16"] = "bf16"


@json_schema_type
class Int4QuantizationConfig(BaseModel):
    """Configuration for 4-bit integer quantization.

    :param type: Must be "int4" to identify this quantization type
    :param scheme: Quantization scheme to use. Defaults to "int4_weight_int8_dynamic_activation"
    """

    type: Literal["int4_mixed"] = "int4_mixed"
    scheme: str | None = "int4_weight_int8_dynamic_activation"


QuantizationConfig = Annotated[
    Bf16QuantizationConfig | Fp8QuantizationConfig | Int4QuantizationConfig,
    Field(discriminator="type"),
]


@json_schema_type
class UserMessage(BaseModel):
    """A message from the user in a chat conversation.

    :param role: Must be "user" to identify this as a user message
    :param content: The content of the message, which can include text and other media
    :param context: (Optional) This field is used internally by Llama Stack to pass RAG context. This field may be removed in the API in the future.
    """

    role: Literal["user"] = "user"
    content: InterleavedContent
    context: InterleavedContent | None = None


@json_schema_type
class SystemMessage(BaseModel):
    """A system message providing instructions or context to the model.

    :param role: Must be "system" to identify this as a system message
    :param content: The content of the "system prompt". If multiple system messages are provided, they are concatenated. The underlying Llama Stack code may also add other system messages (for example, for formatting tool definitions).
    """

    role: Literal["system"] = "system"
    content: InterleavedContent


@json_schema_type
class ToolResponseMessage(BaseModel):
    """A message representing the result of a tool invocation.

    :param role: Must be "tool" to identify this as a tool response
    :param call_id: Unique identifier for the tool call this response is for
    :param content: The response content from the tool
    """

    role: Literal["tool"] = "tool"
    call_id: str
    content: InterleavedContent


@json_schema_type
class CompletionMessage(BaseModel):
    """A message containing the model's (assistant) response in a chat conversation.

    :param role: Must be "assistant" to identify this as the model's response
    :param content: The content of the model's response
    :param stop_reason: Reason why the model stopped generating. Options are:
        - `StopReason.end_of_turn`: The model finished generating the entire response.
        - `StopReason.end_of_message`: The model finished generating but generated a partial response -- usually, a tool call. The user may call the tool and continue the conversation with the tool's response.
        - `StopReason.out_of_tokens`: The model ran out of token budget.
    :param tool_calls: List of tool calls. Each tool call is a ToolCall object.
    """

    role: Literal["assistant"] = "assistant"
    content: InterleavedContent
    stop_reason: StopReason
    tool_calls: list[ToolCall] | None = Field(default_factory=lambda: [])


Message = Annotated[
    UserMessage | SystemMessage | ToolResponseMessage | CompletionMessage,
    Field(discriminator="role"),
]
register_schema(Message, name="Message")


@json_schema_type
class ToolResponse(BaseModel):
    call_id: str
    tool_name: BuiltinTool | str
    content: InterleavedContent
    metadata: dict[str, Any] | None = None

    @field_validator("tool_name", mode="before")
    @classmethod
    def validate_field(cls, v):
        if isinstance(v, str):
            try:
                return BuiltinTool(v)
            except ValueError:
                return v
        return v


class ToolChoice(Enum):
    """Whether tool use is required or automatic. This is a hint to the model which may not be followed. It depends on the Instruction Following capabilities of the model.

    :cvar auto: The model may use tools if it determines that is appropriate.
    :cvar required: The model must use tools.
    :cvar none: The model must not use tools.
    """

    auto = "auto"
    required = "required"
    none = "none"


@json_schema_type
class TokenLogProbs(BaseModel):
    """Log probabilities for generated tokens.

    :param logprobs_by_token: Dictionary mapping tokens to their log probabilities
    """

    logprobs_by_token: dict[str, float]


class ChatCompletionResponseEventType(Enum):
    """Types of events that can occur during chat completion.

    :cvar start: Inference has started
    :cvar complete: Inference is complete and a full response is available
    :cvar progress: Inference is in progress and a partial response is available
    """

    start = "start"
    complete = "complete"
    progress = "progress"


@json_schema_type
class ChatCompletionResponseEvent(BaseModel):
    """An event during chat completion generation.

    :param event_type: Type of the event
    :param delta: Content generated since last event. This can be one or more tokens, or a tool call.
    :param logprobs: Optional log probabilities for generated tokens
    :param stop_reason: Optional reason why generation stopped, if complete
    """

    event_type: ChatCompletionResponseEventType
    delta: ContentDelta
    logprobs: list[TokenLogProbs] | None = None
    stop_reason: StopReason | None = None


class ResponseFormatType(StrEnum):
    """Types of formats for structured (guided) decoding.

    :cvar json_schema: Response should conform to a JSON schema. In a Python SDK, this is often a `pydantic` model.
    :cvar grammar: Response should conform to a BNF grammar
    """

    json_schema = "json_schema"
    grammar = "grammar"


@json_schema_type
class JsonSchemaResponseFormat(BaseModel):
    """Configuration for JSON schema-guided response generation.

    :param type: Must be "json_schema" to identify this format type
    :param json_schema: The JSON schema the response should conform to. In a Python SDK, this is often a `pydantic` model.
    """

    type: Literal[ResponseFormatType.json_schema] = ResponseFormatType.json_schema
    json_schema: dict[str, Any]


@json_schema_type
class GrammarResponseFormat(BaseModel):
    """Configuration for grammar-guided response generation.

    :param type: Must be "grammar" to identify this format type
    :param bnf: The BNF grammar specification the response should conform to
    """

    type: Literal[ResponseFormatType.grammar] = ResponseFormatType.grammar
    bnf: dict[str, Any]


ResponseFormat = Annotated[
    JsonSchemaResponseFormat | GrammarResponseFormat,
    Field(discriminator="type"),
]
register_schema(ResponseFormat, name="ResponseFormat")


# This is an internally used class
class CompletionRequest(BaseModel):
    model: str
    content: InterleavedContent
    sampling_params: SamplingParams | None = Field(default_factory=SamplingParams)
    response_format: ResponseFormat | None = None
    stream: bool | None = False
    logprobs: LogProbConfig | None = None


@json_schema_type
class CompletionResponse(MetricResponseMixin):
    """Response from a completion request.

    :param content: The generated completion text
    :param stop_reason: Reason why generation stopped
    :param logprobs: Optional log probabilities for generated tokens
    """

    content: str
    stop_reason: StopReason
    logprobs: list[TokenLogProbs] | None = None


@json_schema_type
class CompletionResponseStreamChunk(MetricResponseMixin):
    """A chunk of a streamed completion response.

    :param delta: New content generated since last chunk. This can be one or more tokens.
    :param stop_reason: Optional reason why generation stopped, if complete
    :param logprobs: Optional log probabilities for generated tokens
    """

    delta: str
    stop_reason: StopReason | None = None
    logprobs: list[TokenLogProbs] | None = None


class SystemMessageBehavior(Enum):
    """Config for how to override the default system prompt.

    :cvar append: Appends the provided system message to the default system prompt:
        https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/#-function-definitions-in-the-system-prompt-
    :cvar replace: Replaces the default system prompt with the provided system message. The system message can include the string
        '{{function_definitions}}' to indicate where the function definitions should be inserted.
    """

    append = "append"
    replace = "replace"


@json_schema_type
class ToolConfig(BaseModel):
    """Configuration for tool use.

    :param tool_choice: (Optional) Whether tool use is automatic, required, or none. Can also specify a tool name to use a specific tool. Defaults to ToolChoice.auto.
    :param tool_prompt_format: (Optional) Instructs the model how to format tool calls. By default, Llama Stack will attempt to use a format that is best adapted to the model.
        - `ToolPromptFormat.json`: The tool calls are formatted as a JSON object.
        - `ToolPromptFormat.function_tag`: The tool calls are enclosed in a <function=function_name> tag.
        - `ToolPromptFormat.python_list`: The tool calls are output as Python syntax -- a list of function calls.
    :param system_message_behavior: (Optional) Config for how to override the default system prompt.
        - `SystemMessageBehavior.append`: Appends the provided system message to the default system prompt.
        - `SystemMessageBehavior.replace`: Replaces the default system prompt with the provided system message. The system message can include the string
            '{{function_definitions}}' to indicate where the function definitions should be inserted.
    """

    tool_choice: ToolChoice | str | None = Field(default=ToolChoice.auto)
    tool_prompt_format: ToolPromptFormat | None = Field(default=None)
    system_message_behavior: SystemMessageBehavior | None = Field(default=SystemMessageBehavior.append)

    def model_post_init(self, __context: Any) -> None:
        if isinstance(self.tool_choice, str):
            try:
                self.tool_choice = ToolChoice[self.tool_choice]
            except KeyError:
                pass


# This is an internally used class
@json_schema_type
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    sampling_params: SamplingParams | None = Field(default_factory=SamplingParams)

    tools: list[ToolDefinition] | None = Field(default_factory=lambda: [])
    tool_config: ToolConfig | None = Field(default_factory=ToolConfig)

    response_format: ResponseFormat | None = None
    stream: bool | None = False
    logprobs: LogProbConfig | None = None


@json_schema_type
class ChatCompletionResponseStreamChunk(MetricResponseMixin):
    """A chunk of a streamed chat completion response.

    :param event: The event containing the new content
    """

    event: ChatCompletionResponseEvent


@json_schema_type
class ChatCompletionResponse(MetricResponseMixin):
    """Response from a chat completion request.

    :param completion_message: The complete response message
    :param logprobs: Optional log probabilities for generated tokens
    """

    completion_message: CompletionMessage
    logprobs: list[TokenLogProbs] | None = None


@json_schema_type
class EmbeddingsResponse(BaseModel):
    """Response containing generated embeddings.

    :param embeddings: List of embedding vectors, one per input content. Each embedding is a list of floats. The dimensionality of the embedding is model-specific; you can check model metadata using /models/{model_id}
    """

    embeddings: list[list[float]]


@json_schema_type
class OpenAIChatCompletionContentPartTextParam(BaseModel):
    type: Literal["text"] = "text"
    text: str


@json_schema_type
class OpenAIImageURL(BaseModel):
    url: str
    detail: str | None = None


@json_schema_type
class OpenAIChatCompletionContentPartImageParam(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: OpenAIImageURL


@json_schema_type
class OpenAIFileFile(BaseModel):
    file_data: str | None = None
    file_id: str | None = None
    filename: str | None = None


@json_schema_type
class OpenAIFile(BaseModel):
    type: Literal["file"] = "file"
    file: OpenAIFileFile


OpenAIChatCompletionContentPartParam = Annotated[
    OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam | OpenAIFile,
    Field(discriminator="type"),
]
register_schema(OpenAIChatCompletionContentPartParam, name="OpenAIChatCompletionContentPartParam")


OpenAIChatCompletionMessageContent = str | list[OpenAIChatCompletionContentPartParam]

OpenAIChatCompletionTextOnlyMessageContent = str | list[OpenAIChatCompletionContentPartTextParam]


@json_schema_type
class OpenAIUserMessageParam(BaseModel):
    """A message from the user in an OpenAI-compatible chat completion request.

    :param role: Must be "user" to identify this as a user message
    :param content: The content of the message, which can include text and other media
    :param name: (Optional) The name of the user message participant.
    """

    role: Literal["user"] = "user"
    content: OpenAIChatCompletionMessageContent
    name: str | None = None


@json_schema_type
class OpenAISystemMessageParam(BaseModel):
    """A system message providing instructions or context to the model.

    :param role: Must be "system" to identify this as a system message
    :param content: The content of the "system prompt". If multiple system messages are provided, they are concatenated. The underlying Llama Stack code may also add other system messages (for example, for formatting tool definitions).
    :param name: (Optional) The name of the system message participant.
    """

    role: Literal["system"] = "system"
    content: OpenAIChatCompletionTextOnlyMessageContent
    name: str | None = None


@json_schema_type
class OpenAIChatCompletionToolCallFunction(BaseModel):
    name: str | None = None
    arguments: str | None = None


@json_schema_type
class OpenAIChatCompletionToolCall(BaseModel):
    index: int | None = None
    id: str | None = None
    type: Literal["function"] = "function"
    function: OpenAIChatCompletionToolCallFunction | None = None


@json_schema_type
class OpenAIAssistantMessageParam(BaseModel):
    """A message containing the model's (assistant) response in an OpenAI-compatible chat completion request.

    :param role: Must be "assistant" to identify this as the model's response
    :param content: The content of the model's response
    :param name: (Optional) The name of the assistant message participant.
    :param tool_calls: List of tool calls. Each tool call is an OpenAIChatCompletionToolCall object.
    """

    role: Literal["assistant"] = "assistant"
    content: OpenAIChatCompletionTextOnlyMessageContent | None = None
    name: str | None = None
    tool_calls: list[OpenAIChatCompletionToolCall] | None = None


@json_schema_type
class OpenAIToolMessageParam(BaseModel):
    """A message representing the result of a tool invocation in an OpenAI-compatible chat completion request.

    :param role: Must be "tool" to identify this as a tool response
    :param tool_call_id: Unique identifier for the tool call this response is for
    :param content: The response content from the tool
    """

    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: OpenAIChatCompletionTextOnlyMessageContent


@json_schema_type
class OpenAIDeveloperMessageParam(BaseModel):
    """A message from the developer in an OpenAI-compatible chat completion request.

    :param role: Must be "developer" to identify this as a developer message
    :param content: The content of the developer message
    :param name: (Optional) The name of the developer message participant.
    """

    role: Literal["developer"] = "developer"
    content: OpenAIChatCompletionTextOnlyMessageContent
    name: str | None = None


OpenAIMessageParam = Annotated[
    OpenAIUserMessageParam
    | OpenAISystemMessageParam
    | OpenAIAssistantMessageParam
    | OpenAIToolMessageParam
    | OpenAIDeveloperMessageParam,
    Field(discriminator="role"),
]
register_schema(OpenAIMessageParam, name="OpenAIMessageParam")


@json_schema_type
class OpenAIResponseFormatText(BaseModel):
    type: Literal["text"] = "text"


@json_schema_type
class OpenAIJSONSchema(TypedDict, total=False):
    name: str
    description: str | None
    strict: bool | None

    # Pydantic BaseModel cannot be used with a schema param, since it already
    # has one. And, we don't want to alias here because then have to handle
    # that alias when converting to OpenAI params. So, to support schema,
    # we use a TypedDict.
    schema: dict[str, Any] | None


@json_schema_type
class OpenAIResponseFormatJSONSchema(BaseModel):
    type: Literal["json_schema"] = "json_schema"
    json_schema: OpenAIJSONSchema


@json_schema_type
class OpenAIResponseFormatJSONObject(BaseModel):
    type: Literal["json_object"] = "json_object"


OpenAIResponseFormatParam = Annotated[
    OpenAIResponseFormatText | OpenAIResponseFormatJSONSchema | OpenAIResponseFormatJSONObject,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseFormatParam, name="OpenAIResponseFormatParam")


@json_schema_type
class OpenAITopLogProb(BaseModel):
    """The top log probability for a token from an OpenAI-compatible chat completion response.

    :token: The token
    :bytes: (Optional) The bytes for the token
    :logprob: The log probability of the token
    """

    token: str
    bytes: list[int] | None = None
    logprob: float


@json_schema_type
class OpenAITokenLogProb(BaseModel):
    """The log probability for a token from an OpenAI-compatible chat completion response.

    :token: The token
    :bytes: (Optional) The bytes for the token
    :logprob: The log probability of the token
    :top_logprobs: The top log probabilities for the token
    """

    token: str
    bytes: list[int] | None = None
    logprob: float
    top_logprobs: list[OpenAITopLogProb]


@json_schema_type
class OpenAIChoiceLogprobs(BaseModel):
    """The log probabilities for the tokens in the message from an OpenAI-compatible chat completion response.

    :param content: (Optional) The log probabilities for the tokens in the message
    :param refusal: (Optional) The log probabilities for the tokens in the message
    """

    content: list[OpenAITokenLogProb] | None = None
    refusal: list[OpenAITokenLogProb] | None = None


@json_schema_type
class OpenAIChoiceDelta(BaseModel):
    """A delta from an OpenAI-compatible chat completion streaming response.

    :param content: (Optional) The content of the delta
    :param refusal: (Optional) The refusal of the delta
    :param role: (Optional) The role of the delta
    :param tool_calls: (Optional) The tool calls of the delta
    """

    content: str | None = None
    refusal: str | None = None
    role: str | None = None
    tool_calls: list[OpenAIChatCompletionToolCall] | None = None


@json_schema_type
class OpenAIChunkChoice(BaseModel):
    """A chunk choice from an OpenAI-compatible chat completion streaming response.

    :param delta: The delta from the chunk
    :param finish_reason: The reason the model stopped generating
    :param index: The index of the choice
    :param logprobs: (Optional) The log probabilities for the tokens in the message
    """

    delta: OpenAIChoiceDelta
    finish_reason: str
    index: int
    logprobs: OpenAIChoiceLogprobs | None = None


@json_schema_type
class OpenAIChoice(BaseModel):
    """A choice from an OpenAI-compatible chat completion response.

    :param message: The message from the model
    :param finish_reason: The reason the model stopped generating
    :param index: The index of the choice
    :param logprobs: (Optional) The log probabilities for the tokens in the message
    """

    message: OpenAIMessageParam
    finish_reason: str
    index: int
    logprobs: OpenAIChoiceLogprobs | None = None


@json_schema_type
class OpenAIChatCompletion(BaseModel):
    """Response from an OpenAI-compatible chat completion request.

    :param id: The ID of the chat completion
    :param choices: List of choices
    :param object: The object type, which will be "chat.completion"
    :param created: The Unix timestamp in seconds when the chat completion was created
    :param model: The model that was used to generate the chat completion
    """

    id: str
    choices: list[OpenAIChoice]
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str


@json_schema_type
class OpenAIChatCompletionChunk(BaseModel):
    """Chunk from a streaming response to an OpenAI-compatible chat completion request.

    :param id: The ID of the chat completion
    :param choices: List of choices
    :param object: The object type, which will be "chat.completion.chunk"
    :param created: The Unix timestamp in seconds when the chat completion was created
    :param model: The model that was used to generate the chat completion
    """

    id: str
    choices: list[OpenAIChunkChoice]
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str


@json_schema_type
class OpenAICompletionLogprobs(BaseModel):
    """The log probabilities for the tokens in the message from an OpenAI-compatible completion response.

    :text_offset: (Optional) The offset of the token in the text
    :token_logprobs: (Optional) The log probabilities for the tokens
    :tokens: (Optional) The tokens
    :top_logprobs: (Optional) The top log probabilities for the tokens
    """

    text_offset: list[int] | None = None
    token_logprobs: list[float] | None = None
    tokens: list[str] | None = None
    top_logprobs: list[dict[str, float]] | None = None


@json_schema_type
class OpenAICompletionChoice(BaseModel):
    """A choice from an OpenAI-compatible completion response.

    :finish_reason: The reason the model stopped generating
    :text: The text of the choice
    :index: The index of the choice
    :logprobs: (Optional) The log probabilities for the tokens in the choice
    """

    finish_reason: str
    text: str
    index: int
    logprobs: OpenAIChoiceLogprobs | None = None


@json_schema_type
class OpenAICompletion(BaseModel):
    """Response from an OpenAI-compatible completion request.

    :id: The ID of the completion
    :choices: List of choices
    :created: The Unix timestamp in seconds when the completion was created
    :model: The model that was used to generate the completion
    :object: The object type, which will be "text_completion"
    """

    id: str
    choices: list[OpenAICompletionChoice]
    created: int
    model: str
    object: Literal["text_completion"] = "text_completion"


@json_schema_type
class OpenAIEmbeddingData(BaseModel):
    """A single embedding data object from an OpenAI-compatible embeddings response.

    :param object: The object type, which will be "embedding"
    :param embedding: The embedding vector as a list of floats (when encoding_format="float") or as a base64-encoded string (when encoding_format="base64")
    :param index: The index of the embedding in the input list
    """

    object: Literal["embedding"] = "embedding"
    embedding: list[float] | str
    index: int


@json_schema_type
class OpenAIEmbeddingUsage(BaseModel):
    """Usage information for an OpenAI-compatible embeddings response.

    :param prompt_tokens: The number of tokens in the input
    :param total_tokens: The total number of tokens used
    """

    prompt_tokens: int
    total_tokens: int


@json_schema_type
class OpenAIEmbeddingsResponse(BaseModel):
    """Response from an OpenAI-compatible embeddings request.

    :param object: The object type, which will be "list"
    :param data: List of embedding data objects
    :param model: The model that was used to generate the embeddings
    :param usage: Usage information
    """

    object: Literal["list"] = "list"
    data: list[OpenAIEmbeddingData]
    model: str
    usage: OpenAIEmbeddingUsage


class ModelStore(Protocol):
    async def get_model(self, identifier: str) -> Model: ...


class TextTruncation(Enum):
    """Config for how to truncate text for embedding when text is longer than the model's max sequence length. Start and End semantics depend on whether the language is left-to-right or right-to-left.

    :cvar none: No truncation (default). If the text is longer than the model's max sequence length, you will get an error.
    :cvar start: Truncate from the start
    :cvar end: Truncate from the end
    """

    none = "none"
    start = "start"
    end = "end"


class EmbeddingTaskType(Enum):
    """How is the embedding being used? This is only supported by asymmetric embedding models.

    :cvar query: Used for a query for semantic search.
    :cvar document: Used at indexing time when ingesting documents.
    """

    query = "query"
    document = "document"


@json_schema_type
class BatchCompletionResponse(BaseModel):
    batch: list[CompletionResponse]


@json_schema_type
class BatchChatCompletionResponse(BaseModel):
    batch: list[ChatCompletionResponse]


class OpenAICompletionWithInputMessages(OpenAIChatCompletion):
    input_messages: list[OpenAIMessageParam]


@json_schema_type
class ListOpenAIChatCompletionResponse(BaseModel):
    data: list[OpenAICompletionWithInputMessages]
    has_more: bool
    first_id: str
    last_id: str
    object: Literal["list"] = "list"


@runtime_checkable
@trace_protocol
class InferenceProvider(Protocol):
    """
    This protocol defines the interface that should be implemented by all inference providers.
    """

    API_NAMESPACE: str = "Inference"

    model_store: ModelStore | None = None

    @webmethod(route="/inference/completion", method="POST")
    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
    ) -> CompletionResponse | AsyncIterator[CompletionResponseStreamChunk]:
        """Generate a completion for the given content using the specified model.

        :param model_id: The identifier of the model to use. The model must be registered with Llama Stack and available via the /models endpoint.
        :param content: The content to generate a completion for.
        :param sampling_params: (Optional) Parameters to control the sampling strategy.
        :param response_format: (Optional) Grammar specification for guided (structured) decoding.
        :param stream: (Optional) If True, generate an SSE event stream of the response. Defaults to False.
        :param logprobs: (Optional) If specified, log probabilities for each token position will be returned.
        :returns: If stream=False, returns a CompletionResponse with the full completion.
                 If stream=True, returns an SSE event stream of CompletionResponseStreamChunk.
        """
        ...

    @webmethod(route="/inference/batch-completion", method="POST", experimental=True)
    async def batch_completion(
        self,
        model_id: str,
        content_batch: list[InterleavedContent],
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        logprobs: LogProbConfig | None = None,
    ) -> BatchCompletionResponse:
        """Generate completions for a batch of content using the specified model.

        :param model_id: The identifier of the model to use. The model must be registered with Llama Stack and available via the /models endpoint.
        :param content_batch: The content to generate completions for.
        :param sampling_params: (Optional) Parameters to control the sampling strategy.
        :param response_format: (Optional) Grammar specification for guided (structured) decoding.
        :param logprobs: (Optional) If specified, log probabilities for each token position will be returned.
        :returns: A BatchCompletionResponse with the full completions.
        """
        raise NotImplementedError("Batch completion is not implemented")

    @webmethod(route="/inference/chat-completion", method="POST")
    async def chat_completion(
        self,
        model_id: str,
        messages: list[Message],
        sampling_params: SamplingParams | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = ToolChoice.auto,
        tool_prompt_format: ToolPromptFormat | None = None,
        response_format: ResponseFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
        tool_config: ToolConfig | None = None,
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionResponseStreamChunk]:
        """Generate a chat completion for the given messages using the specified model.

        :param model_id: The identifier of the model to use. The model must be registered with Llama Stack and available via the /models endpoint.
        :param messages: List of messages in the conversation.
        :param sampling_params: Parameters to control the sampling strategy.
        :param tools: (Optional) List of tool definitions available to the model.
        :param tool_choice: (Optional) Whether tool use is required or automatic. Defaults to ToolChoice.auto.
            .. deprecated::
               Use tool_config instead.
        :param tool_prompt_format: (Optional) Instructs the model how to format tool calls. By default, Llama Stack will attempt to use a format that is best adapted to the model.
            - `ToolPromptFormat.json`: The tool calls are formatted as a JSON object.
            - `ToolPromptFormat.function_tag`: The tool calls are enclosed in a <function=function_name> tag.
            - `ToolPromptFormat.python_list`: The tool calls are output as Python syntax -- a list of function calls.
            .. deprecated::
               Use tool_config instead.
        :param response_format: (Optional) Grammar specification for guided (structured) decoding. There are two options:
            - `ResponseFormat.json_schema`: The grammar is a JSON schema. Most providers support this format.
            - `ResponseFormat.grammar`: The grammar is a BNF grammar. This format is more flexible, but not all providers support it.
        :param stream: (Optional) If True, generate an SSE event stream of the response. Defaults to False.
        :param logprobs: (Optional) If specified, log probabilities for each token position will be returned.
        :param tool_config: (Optional) Configuration for tool use.
        :returns: If stream=False, returns a ChatCompletionResponse with the full completion.
                 If stream=True, returns an SSE event stream of ChatCompletionResponseStreamChunk.
        """
        ...

    @webmethod(route="/inference/batch-chat-completion", method="POST", experimental=True)
    async def batch_chat_completion(
        self,
        model_id: str,
        messages_batch: list[list[Message]],
        sampling_params: SamplingParams | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_config: ToolConfig | None = None,
        response_format: ResponseFormat | None = None,
        logprobs: LogProbConfig | None = None,
    ) -> BatchChatCompletionResponse:
        """Generate chat completions for a batch of messages using the specified model.

        :param model_id: The identifier of the model to use. The model must be registered with Llama Stack and available via the /models endpoint.
        :param messages_batch: The messages to generate completions for.
        :param sampling_params: (Optional) Parameters to control the sampling strategy.
        :param tools: (Optional) List of tool definitions available to the model.
        :param tool_config: (Optional) Configuration for tool use.
        :param response_format: (Optional) Grammar specification for guided (structured) decoding.
        :param logprobs: (Optional) If specified, log probabilities for each token position will be returned.
        :returns: A BatchChatCompletionResponse with the full completions.
        """
        raise NotImplementedError("Batch chat completion is not implemented")

    @webmethod(route="/inference/embeddings", method="POST")
    async def embeddings(
        self,
        model_id: str,
        contents: list[str] | list[InterleavedContentItem],
        text_truncation: TextTruncation | None = TextTruncation.none,
        output_dimension: int | None = None,
        task_type: EmbeddingTaskType | None = None,
    ) -> EmbeddingsResponse:
        """Generate embeddings for content pieces using the specified model.

        :param model_id: The identifier of the model to use. The model must be an embedding model registered with Llama Stack and available via the /models endpoint.
        :param contents: List of contents to generate embeddings for. Each content can be a string or an InterleavedContentItem (and hence can be multimodal). The behavior depends on the model and provider. Some models may only support text.
        :param output_dimension: (Optional) Output dimensionality for the embeddings. Only supported by Matryoshka models.
        :param text_truncation: (Optional) Config for how to truncate text for embedding when text is longer than the model's max sequence length.
        :param task_type: (Optional) How is the embedding being used? This is only supported by asymmetric embedding models.
        :returns: An array of embeddings, one for each content. Each embedding is a list of floats. The dimensionality of the embedding is model-specific; you can check model metadata using /models/{model_id}.
        """
        ...

    @webmethod(route="/openai/v1/completions", method="POST")
    async def openai_completion(
        self,
        # Standard OpenAI completion parameters
        model: str,
        prompt: str | list[str] | list[int] | list[list[int]],
        best_of: int | None = None,
        echo: bool | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream: bool | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        user: str | None = None,
        # vLLM-specific parameters
        guided_choice: list[str] | None = None,
        prompt_logprobs: int | None = None,
        # for fill-in-the-middle type completion
        suffix: str | None = None,
    ) -> OpenAICompletion:
        """Generate an OpenAI-compatible completion for the given prompt using the specified model.

        :param model: The identifier of the model to use. The model must be registered with Llama Stack and available via the /models endpoint.
        :param prompt: The prompt to generate a completion for.
        :param best_of: (Optional) The number of completions to generate.
        :param echo: (Optional) Whether to echo the prompt.
        :param frequency_penalty: (Optional) The penalty for repeated tokens.
        :param logit_bias: (Optional) The logit bias to use.
        :param logprobs: (Optional) The log probabilities to use.
        :param max_tokens: (Optional) The maximum number of tokens to generate.
        :param n: (Optional) The number of completions to generate.
        :param presence_penalty: (Optional) The penalty for repeated tokens.
        :param seed: (Optional) The seed to use.
        :param stop: (Optional) The stop tokens to use.
        :param stream: (Optional) Whether to stream the response.
        :param stream_options: (Optional) The stream options to use.
        :param temperature: (Optional) The temperature to use.
        :param top_p: (Optional) The top p to use.
        :param user: (Optional) The user to use.
        :param suffix: (Optional) The suffix that should be appended to the completion.
        :returns: An OpenAICompletion.
        """
        ...

    @webmethod(route="/openai/v1/chat/completions", method="POST")
    async def openai_chat_completion(
        self,
        model: str,
        messages: list[OpenAIMessageParam],
        frequency_penalty: float | None = None,
        function_call: str | dict[str, Any] | None = None,
        functions: list[dict[str, Any]] | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
        presence_penalty: float | None = None,
        response_format: OpenAIResponseFormatParam | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream: bool | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        top_logprobs: int | None = None,
        top_p: float | None = None,
        user: str | None = None,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        """Generate an OpenAI-compatible chat completion for the given messages using the specified model.

        :param model: The identifier of the model to use. The model must be registered with Llama Stack and available via the /models endpoint.
        :param messages: List of messages in the conversation.
        :param frequency_penalty: (Optional) The penalty for repeated tokens.
        :param function_call: (Optional) The function call to use.
        :param functions: (Optional) List of functions to use.
        :param logit_bias: (Optional) The logit bias to use.
        :param logprobs: (Optional) The log probabilities to use.
        :param max_completion_tokens: (Optional) The maximum number of tokens to generate.
        :param max_tokens: (Optional) The maximum number of tokens to generate.
        :param n: (Optional) The number of completions to generate.
        :param parallel_tool_calls: (Optional) Whether to parallelize tool calls.
        :param presence_penalty: (Optional) The penalty for repeated tokens.
        :param response_format: (Optional) The response format to use.
        :param seed: (Optional) The seed to use.
        :param stop: (Optional) The stop tokens to use.
        :param stream: (Optional) Whether to stream the response.
        :param stream_options: (Optional) The stream options to use.
        :param temperature: (Optional) The temperature to use.
        :param tool_choice: (Optional) The tool choice to use.
        :param tools: (Optional) The tools to use.
        :param top_logprobs: (Optional) The top log probabilities to use.
        :param top_p: (Optional) The top p to use.
        :param user: (Optional) The user to use.
        :returns: An OpenAIChatCompletion.
        """
        ...

    @webmethod(route="/openai/v1/embeddings", method="POST")
    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        """Generate OpenAI-compatible embeddings for the given input using the specified model.

        :param model: The identifier of the model to use. The model must be an embedding model registered with Llama Stack and available via the /models endpoint.
        :param input: Input text to embed, encoded as a string or array of strings. To embed multiple inputs in a single request, pass an array of strings.
        :param encoding_format: (Optional) The format to return the embeddings in. Can be either "float" or "base64". Defaults to "float".
        :param dimensions: (Optional) The number of dimensions the resulting output embeddings should have. Only supported in text-embedding-3 and later models.
        :param user: (Optional) A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
        :returns: An OpenAIEmbeddingsResponse containing the embeddings.
        """
        ...


class Inference(InferenceProvider):
    """Llama Stack Inference API for generating completions, chat completions, and embeddings.

    This API provides the raw interface to the underlying models. Two kinds of models are supported:
    - LLM models: these models generate "raw" and "chat" (conversational) completions.
    - Embedding models: these models generate embeddings to be used for semantic search.
    """

    @webmethod(route="/openai/v1/chat/completions", method="GET")
    async def list_chat_completions(
        self,
        after: str | None = None,
        limit: int | None = 20,
        model: str | None = None,
        order: Order | None = Order.desc,
    ) -> ListOpenAIChatCompletionResponse:
        """List all chat completions.

        :param after: The ID of the last chat completion to return.
        :param limit: The maximum number of chat completions to return.
        :param model: The model to filter by.
        :param order: The order to sort the chat completions by: "asc" or "desc". Defaults to "desc".
        :returns: A ListOpenAIChatCompletionResponse.
        """
        raise NotImplementedError("List chat completions is not implemented")

    @webmethod(route="/openai/v1/chat/completions/{completion_id}", method="GET")
    async def get_chat_completion(self, completion_id: str) -> OpenAICompletionWithInputMessages:
        """Describe a chat completion by its ID.

        :param completion_id: ID of the chat completion.
        :returns: A OpenAICompletionWithInputMessages.
        """
        raise NotImplementedError("Get chat completion is not implemented")
