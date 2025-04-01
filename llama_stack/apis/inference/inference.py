# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel, Field, field_validator
from typing_extensions import Annotated

from llama_stack.apis.common.content_types import ContentDelta, InterleavedContent, InterleavedContentItem
from llama_stack.apis.models import Model
from llama_stack.apis.telemetry.telemetry import MetricResponseMixin
from llama_stack.models.llama.datatypes import (
    BuiltinTool,
    SamplingParams,
    StopReason,
    ToolCall,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol
from llama_stack.schema_utils import json_schema_type, register_schema, webmethod


class LogProbConfig(BaseModel):
    """

    :param top_k: How many tokens (for each position) to return log probabilities for.
    """

    top_k: Optional[int] = 0


class QuantizationType(Enum):
    """Type of model quantization to run inference with.

    :cvar bf16: BFloat16 typically this means _no_ quantization
    :cvar fp8: 8-bit floating point quantization
    :cvar int4: 4-bit integer quantization
    """

    bf16 = "bf16"
    fp8 = "fp8"
    int4 = "int4"


@json_schema_type
class Fp8QuantizationConfig(BaseModel):
    type: Literal["fp8"] = "fp8"


@json_schema_type
class Bf16QuantizationConfig(BaseModel):
    type: Literal["bf16"] = "bf16"


@json_schema_type
class Int4QuantizationConfig(BaseModel):
    """Configuration for 4-bit integer quantization.

    :param type: Must be "int4" to identify this quantization type
    :param scheme: Quantization scheme to use. Defaults to "int4_weight_int8_dynamic_activation"
    """

    type: Literal["int4"] = "int4"
    scheme: Optional[str] = "int4_weight_int8_dynamic_activation"


QuantizationConfig = Annotated[
    Union[Bf16QuantizationConfig, Fp8QuantizationConfig, Int4QuantizationConfig],
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
    context: Optional[InterleavedContent] = None


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
    tool_calls: Optional[List[ToolCall]] = Field(default_factory=list)


Message = Annotated[
    Union[
        UserMessage,
        SystemMessage,
        ToolResponseMessage,
        CompletionMessage,
    ],
    Field(discriminator="role"),
]
register_schema(Message, name="Message")


@json_schema_type
class ToolResponse(BaseModel):
    call_id: str
    tool_name: Union[BuiltinTool, str]
    content: InterleavedContent
    metadata: Optional[Dict[str, Any]] = None

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

    logprobs_by_token: Dict[str, float]


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
    logprobs: Optional[List[TokenLogProbs]] = None
    stop_reason: Optional[StopReason] = None


class ResponseFormatType(Enum):
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

    type: Literal[ResponseFormatType.json_schema.value] = ResponseFormatType.json_schema.value
    json_schema: Dict[str, Any]


@json_schema_type
class GrammarResponseFormat(BaseModel):
    """Configuration for grammar-guided response generation.

    :param type: Must be "grammar" to identify this format type
    :param bnf: The BNF grammar specification the response should conform to
    """

    type: Literal[ResponseFormatType.grammar.value] = ResponseFormatType.grammar.value
    bnf: Dict[str, Any]


ResponseFormat = Annotated[
    Union[JsonSchemaResponseFormat, GrammarResponseFormat],
    Field(discriminator="type"),
]
register_schema(ResponseFormat, name="ResponseFormat")


# This is an internally used class
class CompletionRequest(BaseModel):
    model: str
    content: InterleavedContent
    sampling_params: Optional[SamplingParams] = Field(default_factory=SamplingParams)
    response_format: Optional[ResponseFormat] = None
    stream: Optional[bool] = False
    logprobs: Optional[LogProbConfig] = None


@json_schema_type
class CompletionResponse(MetricResponseMixin):
    """Response from a completion request.

    :param content: The generated completion text
    :param stop_reason: Reason why generation stopped
    :param logprobs: Optional log probabilities for generated tokens
    """

    content: str
    stop_reason: StopReason
    logprobs: Optional[List[TokenLogProbs]] = None


@json_schema_type
class CompletionResponseStreamChunk(MetricResponseMixin):
    """A chunk of a streamed completion response.

    :param delta: New content generated since last chunk. This can be one or more tokens.
    :param stop_reason: Optional reason why generation stopped, if complete
    :param logprobs: Optional log probabilities for generated tokens
    """

    delta: str
    stop_reason: Optional[StopReason] = None
    logprobs: Optional[List[TokenLogProbs]] = None


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

    tool_choice: Optional[ToolChoice | str] = Field(default=ToolChoice.auto)
    tool_prompt_format: Optional[ToolPromptFormat] = Field(default=None)
    system_message_behavior: Optional[SystemMessageBehavior] = Field(default=SystemMessageBehavior.append)

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
    messages: List[Message]
    sampling_params: Optional[SamplingParams] = Field(default_factory=SamplingParams)

    tools: Optional[List[ToolDefinition]] = Field(default_factory=list)
    tool_config: Optional[ToolConfig] = Field(default_factory=ToolConfig)

    response_format: Optional[ResponseFormat] = None
    stream: Optional[bool] = False
    logprobs: Optional[LogProbConfig] = None


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
    logprobs: Optional[List[TokenLogProbs]] = None


@json_schema_type
class EmbeddingsResponse(BaseModel):
    """Response containing generated embeddings.

    :param embeddings: List of embedding vectors, one per input content. Each embedding is a list of floats. The dimensionality of the embedding is model-specific; you can check model metadata using /models/{model_id}
    """

    embeddings: List[List[float]]


class ModelStore(Protocol):
    def get_model(self, identifier: str) -> Model: ...


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


@runtime_checkable
@trace_protocol
class Inference(Protocol):
    """Llama Stack Inference API for generating completions, chat completions, and embeddings.

    This API provides the raw interface to the underlying models. Two kinds of models are supported:
    - LLM models: these models generate "raw" and "chat" (conversational) completions.
    - Embedding models: these models generate embeddings to be used for semantic search.
    """

    model_store: ModelStore

    @webmethod(route="/inference/completion", method="POST")
    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponseStreamChunk]]:
        """Generate a completion for the given content using the specified model.

        :param model_id: The identifier of the model to use. The model must be registered with Llama Stack and available via the /models endpoint.
        :param content: The content to generate a completion for
        :param sampling_params: (Optional) Parameters to control the sampling strategy
        :param response_format: (Optional) Grammar specification for guided (structured) decoding
        :param stream: (Optional) If True, generate an SSE event stream of the response. Defaults to False.
        :param logprobs: (Optional) If specified, log probabilities for each token position will be returned.
        :returns: If stream=False, returns a CompletionResponse with the full completion.
                 If stream=True, returns an SSE event stream of CompletionResponseStreamChunk
        """
        ...

    @webmethod(route="/inference/chat-completion", method="POST")
    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
        tool_config: Optional[ToolConfig] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]]:
        """Generate a chat completion for the given messages using the specified model.

        :param model_id: The identifier of the model to use. The model must be registered with Llama Stack and available via the /models endpoint.
        :param messages: List of messages in the conversation
        :param sampling_params: Parameters to control the sampling strategy
        :param tools: (Optional) List of tool definitions available to the model
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
                 If stream=True, returns an SSE event stream of ChatCompletionResponseStreamChunk
        """
        ...

    @webmethod(route="/inference/embeddings", method="POST")
    async def embeddings(
        self,
        model_id: str,
        contents: List[str] | List[InterleavedContentItem],
        text_truncation: Optional[TextTruncation] = TextTruncation.none,
        output_dimension: Optional[int] = None,
        task_type: Optional[EmbeddingTaskType] = None,
    ) -> EmbeddingsResponse:
        """Generate embeddings for content pieces using the specified model.

        :param model_id: The identifier of the model to use. The model must be an embedding model registered with Llama Stack and available via the /models endpoint.
        :param contents: List of contents to generate embeddings for. Each content can be a string or an InterleavedContentItem (and hence can be multimodal). The behavior depends on the model and provider. Some models may only support text.
        :param output_dimension: (Optional) Output dimensionality for the embeddings. Only supported by Matryoshka models.
        :param text_truncation: (Optional) Config for how to truncate text for embedding when text is longer than the model's max sequence length.
        :param task_type: (Optional) How is the embedding being used? This is only supported by asymmetric embedding models.
        :returns: An array of embeddings, one for each content. Each embedding is a list of floats. The dimensionality of the embedding is model-specific; you can check model metadata using /models/{model_id}
        """
        ...
