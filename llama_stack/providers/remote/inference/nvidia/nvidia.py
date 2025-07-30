# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
import warnings
from collections.abc import AsyncIterator

from openai import APIConnectionError, BadRequestError, AsyncOpenAI

from llama_stack.apis.common.content_types import (
    InterleavedContent,
    InterleavedContentItem,
    TextContentItem,
)
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChunk,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseStreamChunk,
    EmbeddingsResponse,
    EmbeddingTaskType,
    Inference,
    LogProbConfig,
    Message,
    ModelStore,
    ResponseFormat,
    SamplingParams,
    TextTruncation,
    ToolChoice,
    ToolConfig,
)
from llama_stack.apis.models import Model, ModelType
from llama_stack.models.llama.datatypes import ToolDefinition, ToolPromptFormat
from llama_stack.providers.datatypes import (
    HealthResponse,
    HealthStatus,
    ModelsProtocolPrivate,
)
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
)
from llama_stack.providers.utils.inference.openai_compat import (
    convert_openai_chat_completion_choice,
    convert_openai_chat_completion_stream,
)
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin
from llama_stack.providers.utils.inference.prompt_adapter import content_has_media

from . import NVIDIAConfig
from .models import MODEL_ENTRIES
from .openai_utils import (
    convert_chat_completion_request,
    convert_completion_request,
    convert_openai_completion_choice,
    convert_openai_completion_stream,
)
from .utils import _is_nvidia_hosted

logger = logging.getLogger(__name__)


class NVIDIAInferenceAdapter(OpenAIMixin, Inference, ModelRegistryHelper, ModelsProtocolPrivate):
    """
    NVIDIA Inference Adapter for Llama Stack.

    Note: The inheritance order is important here. OpenAIMixin must come before
    ModelRegistryHelper to ensure that OpenAIMixin.check_model_availability()
    is used instead of ModelRegistryHelper.check_model_availability(). It also
    must come before Inference to ensure that OpenAIMixin methods are available
    in the Inference interface.

    - OpenAIMixin.check_model_availability() queries the NVIDIA API to check if a model exists
    - ModelRegistryHelper.check_model_availability() just returns False and shows a warning
    """

    # automatically set by the resolver when instantiating the provider
    __provider_id__: str
    model_store: ModelStore | None = None

    def __init__(self, config: NVIDIAConfig) -> None:
        # TODO(mf): filter by available models
        ModelRegistryHelper.__init__(self, model_entries=MODEL_ENTRIES)

        logger.info(f"Initializing NVIDIAInferenceAdapter({config.url})...")

        if _is_nvidia_hosted(config):
            if not config.api_key:
                raise RuntimeError(
                    "API key is required for hosted NVIDIA NIM. Either provide an API key or use a self-hosted NIM."
                )
        # elif self._config.api_key:
        #
        # we don't raise this warning because a user may have deployed their
        # self-hosted NIM with an API key requirement.
        #
        #     warnings.warn(
        #         "API key is not required for self-hosted NVIDIA NIM. "
        #         "Consider removing the api_key from the configuration."
        #     )

        self._config = config
        self._client = None

    def get_api_key(self) -> str:
        """
        Get the API key for OpenAI mixin.

        :return: The NVIDIA API key
        """
        return self._config.api_key.get_secret_value() if self._config.api_key else "NO KEY"

    def get_base_url(self) -> str:
        """
        Get the base URL for OpenAI mixin.

        :return: The NVIDIA API base URL
        """
        return f"{self._config.url}/v1" if self._config.append_api_version else self._config.url

    @property
    def client(self):
        """
        Get the OpenAI client.
        
        :return: The OpenAI client
        """
        self._lazy_initialize_client()
        return self._client
        
    def _lazy_initialize_client(self):
        """
        Initialize the OpenAI client if it hasn't been initialized yet.
        """
        if self._client is not None:
            return

        logger.info(f"Initializing NVIDIA client with base_url={self.get_base_url()}")
        self._client = AsyncOpenAI(
            base_url=self.get_base_url(),
            api_key=self.get_api_key(),
        )

    async def initialize(self) -> None:
        """
        Initialize the NVIDIA adapter.
        """
        if not self._config.url:
            raise ValueError(
                "You must provide a URL in run.yaml (or via the NVIDIA_BASE_URL environment variable) to use NVIDIA NIM."
            )

    async def should_refresh_models(self) -> bool:
        """
        Determine if models should be refreshed.
        
        :return: True if models should be refreshed, False otherwise
        """
        # Always refresh models to ensure we have the latest available models
        return True

    async def list_models(self) -> list[Model] | None:
        """
        List all models available from the NVIDIA API.
        
        :return: A list of available models
        """
        self._lazy_initialize_client()
        models = []
        try:
            async for m in self.client.models.list():
                # Determine model type based on model ID or capabilities
                # This is a simple heuristic and might need refinement
                model_type = ModelType.llm
                if "embed" in m.id.lower():
                    model_type = ModelType.embedding
                
                models.append(
                    Model(
                        identifier=m.id,
                        provider_resource_id=m.id,
                        provider_id=self.__provider_id__,
                        metadata={},
                        model_type=model_type,
                    )
                )
            return models
        except Exception as e:
            logger.warning(f"Failed to list models from NVIDIA API: {e}")
            return None

    async def register_model(self, model: Model) -> Model:
        """
        Register a model with the NVIDIA adapter.
        
        :param model: The model to register
        :return: The registered model
        """
        self._lazy_initialize_client()
        try:
            # First try to register using the static model entries
            model = await ModelRegistryHelper.register_model(self, model)
        except ValueError:
            pass  # Ignore statically unknown model, will check live listing
        
        try:
            # Check if the model is available on the NVIDIA server
            available_models = [m.id async for m in self.client.models.list()]
            if model.provider_resource_id not in available_models:
                raise ValueError(
                    f"Model {model.provider_resource_id} is not being served by NVIDIA NIM. "
                    f"Available models: {', '.join(available_models)}"
                )
        except APIConnectionError as e:
            raise ValueError(
                f"Failed to connect to NVIDIA NIM at {self._config.url}. Please check if NVIDIA NIM is running and accessible at that URL."
            ) from e
        
        return model

    async def unregister_model(self, model_id: str) -> None:
        """
        Unregister a model from the NVIDIA adapter.
        
        :param model_id: The ID of the model to unregister
        """
        pass

    async def health(self) -> HealthResponse:
        """
        Performs a health check by verifying connectivity to the remote NVIDIA NIM server.
        This method is used by the Provider API to verify
        that the service is running correctly.
        
        :return: A HealthResponse object indicating the health status
        """
        try:
            client = AsyncOpenAI(
                base_url=self.get_base_url(),
                api_key=self.get_api_key(),
            ) if self._client is None else self._client
            _ = [m async for m in client.models.list()]  # Ensure the client is initialized
            return HealthResponse(status=HealthStatus.OK)
        except Exception as e:
            return HealthResponse(status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}")

    async def _get_model(self, model_id: str) -> Model:
        """
        Get a model by ID.
        
        :param model_id: The ID of the model to get
        :return: The model
        """
        if not self.model_store:
            raise ValueError("Model store not set")
        return await self.model_store.get_model(model_id)

    async def shutdown(self) -> None:
        """
        Shutdown the NVIDIA adapter.
        """
        pass

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
        suffix: str | None = None,
    ) -> CompletionResponse | AsyncIterator[CompletionResponseStreamChunk]:
        if sampling_params is None:
            sampling_params = SamplingParams()
        if content_has_media(content):
            raise NotImplementedError("Media is not supported")

        # ToDo: check health of NeMo endpoints and enable this
        # removing this health check as NeMo customizer endpoint health check is returning 404
        # await check_health(self._config)  # this raises errors

        self._lazy_initialize_client()
        provider_model_id = await self._get_provider_model_id(model_id)
        request = convert_completion_request(
            request=CompletionRequest(
                model=provider_model_id,
                content=content,
                sampling_params=sampling_params,
                response_format=response_format,
                stream=stream,
                logprobs=logprobs,
                suffix=suffix,
            ),
            n=1,
        )

        try:
            response = await self.client.completions.create(**request)
        except APIConnectionError as e:
            raise ConnectionError(f"Failed to connect to NVIDIA NIM at {self._config.url}: {e}") from e

        if stream:
            return convert_openai_completion_stream(response)
        else:
            # we pass n=1 to get only one completion
            return convert_openai_completion_choice(response.choices[0])

    async def embeddings(
        self,
        model_id: str,
        contents: list[str] | list[InterleavedContentItem],
        text_truncation: TextTruncation | None = TextTruncation.none,
        output_dimension: int | None = None,
        task_type: EmbeddingTaskType | None = None,
    ) -> EmbeddingsResponse:
        if any(content_has_media(content) for content in contents):
            raise NotImplementedError("Media is not supported")

        #
        # Llama Stack: contents = list[str] | list[InterleavedContentItem]
        #  ->
        # OpenAI: input = str | list[str]
        #
        # we can ignore str and always pass list[str] to OpenAI
        #
        self._lazy_initialize_client()
        flat_contents = [content.text if isinstance(content, TextContentItem) else content for content in contents]
        input = [content.text if isinstance(content, TextContentItem) else content for content in flat_contents]
        provider_model_id = await self._get_provider_model_id(model_id)

        extra_body = {}

        if text_truncation is not None:
            text_truncation_options = {
                TextTruncation.none: "NONE",
                TextTruncation.end: "END",
                TextTruncation.start: "START",
            }
            extra_body["truncate"] = text_truncation_options[text_truncation]

        if output_dimension is not None:
            extra_body["dimensions"] = output_dimension

        if task_type is not None:
            task_type_options = {
                EmbeddingTaskType.document: "passage",
                EmbeddingTaskType.query: "query",
            }
            extra_body["input_type"] = task_type_options[task_type]

        try:
            response = await self.client.embeddings.create(
                model=provider_model_id,
                input=input,
                extra_body=extra_body,
            )
        except BadRequestError as e:
            raise ValueError(f"Failed to get embeddings: {e}") from e

        #
        # OpenAI: CreateEmbeddingResponse(data=[Embedding(embedding=list[float], ...)], ...)
        #  ->
        # Llama Stack: EmbeddingsResponse(embeddings=list[list[float]])
        #
        return EmbeddingsResponse(embeddings=[embedding.embedding for embedding in response.data])

    async def chat_completion(
        self,
        model_id: str,
        messages: list[Message],
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = ToolChoice.auto,
        tool_prompt_format: ToolPromptFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
        tool_config: ToolConfig | None = None,
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionResponseStreamChunk]:
        if sampling_params is None:
            sampling_params = SamplingParams()
        if tool_prompt_format:
            warnings.warn("tool_prompt_format is not supported by NVIDIA NIM, ignoring", stacklevel=2)

        # await check_health(self._config)  # this raises errors

        self._lazy_initialize_client()
        provider_model_id = await self._get_provider_model_id(model_id)
        request = await convert_chat_completion_request(
            request=ChatCompletionRequest(
                model=provider_model_id,
                messages=messages,
                sampling_params=sampling_params,
                response_format=response_format,
                tools=tools,
                stream=stream,
                logprobs=logprobs,
                tool_config=tool_config,
            ),
            n=1,
        )

        try:
            response = await self.client.chat.completions.create(**request)
        except APIConnectionError as e:
            raise ConnectionError(f"Failed to connect to NVIDIA NIM at {self._config.url}: {e}") from e

        if stream:
            return convert_openai_chat_completion_stream(response, enable_incremental_tool_calls=False)
        else:
            # we pass n=1 to get only one completion
            return convert_openai_chat_completion_choice(response.choices[0])
