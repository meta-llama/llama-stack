# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from typing import Any

import httpx2
from typesafe_sdk import AsyncTypeSafeClient, TypeSafeAPIError

from ogx.core.request_headers import NeedsRequestProviderData
from ogx.log import get_logger
from ogx.providers.utils.inference.http_client import build_network_client_kwargs
from ogx_api import (
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicCreateMessageRequest,
    AnthropicMessageResponse,
    InferenceProvider,
    Model,
    ModelsProtocolPrivate,
    ModelType,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionWithReasoning,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
    RerankResponse,
    SystemOneRequest,
    SystemOneResponse,
)
from ogx_api.inference import RerankRequest
from ogx_api.messages.models import AnthropicStreamEvent

from .config import TypeSafeConfig

logger = get_logger(name=__name__, category="inference::typesafe")

_UNIMPLEMENTED = "The TypeSafe provider only serves SystemOne (judgment) models and does not support {operation}."


class TypeSafeProviderError(Exception):
    """A general upstream TypeSafe failure, carrying the upstream HTTP status code.

    The provider translates the SDK's typed errors into this so the core layer can surface the
    upstream status via the generic ``status_code`` path (core.exceptions.translation) without
    importing the SDK.
    """

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _to_provider_error(exc: TypeSafeAPIError) -> TypeSafeProviderError:
    # Surface 4xx as-is; map 5xx (or a missing status) to a bad-gateway, signalling an upstream
    # failure rather than an OGX error.
    status = exc.status
    if status is None or status >= 500:
        status = httpx2.codes.BAD_GATEWAY
    return TypeSafeProviderError(status, str(exc))


class TypeSafeInferenceAdapter(InferenceProvider, ModelsProtocolPrivate, NeedsRequestProviderData):
    """Inference adapter for TypeSafe's SystemOne (judgment/decision) models.

    Uses the official ``typesafe_sdk`` client. Judgment calls go through
    ``client.system_one`` with OGX's ``SystemOneResponse`` wire model as the response model,
    so the raw TypeSafe response body is validated straight into the OGX wire format. The SDK
    sends the ``X-TypeSafe-SDK`` header on every request, which the models router uses to serve
    the TypeSafe-shaped ``GET /v1/models`` payload.
    """

    config: TypeSafeConfig

    __provider_id__: str

    def __init__(self, config: TypeSafeConfig) -> None:
        self.config = config

    async def initialize(self) -> None:
        # A client is built and closed per request (see ``_build_client``); there is nothing
        # to warm up here. A missing API key surfaces only when a call is attempted.
        pass

    async def shutdown(self) -> None:
        # Clients are closed after each request in the caller's ``finally``; nothing persists.
        pass

    def _get_api_key(self) -> str | None:
        api_key: str | None = None
        if self.config.auth_credential is not None:
            api_key = self.config.auth_credential.get_secret_value()
        provider_data = self.get_request_provider_data()
        if provider_data is not None and provider_data.typesafe_api_key is not None:
            api_key = provider_data.typesafe_api_key.get_secret_value()
        return api_key

    def _get_base_url(self) -> str:
        return str(self.config.base_url).rstrip("/")

    def _build_http_client(self) -> httpx2.AsyncClient | None:
        if self.config.network is None:
            return None
        kwargs = build_network_client_kwargs(self.config.network)
        return httpx2.AsyncClient(**kwargs)

    def _build_client(self) -> AsyncTypeSafeClient:
        """Build a fresh ``AsyncTypeSafeClient`` for a single request.

        A new client (and its httpx2 transport) is created per call and closed by the caller
        once the call completes, so there is no shared mutable client to race on or leak. The
        SDK closes a supplied http_client on ``aclose``. The SDK requires a non-empty API key at
        construction, so a missing key raises here (surfacing as a clear error at call time)
        rather than at registration.
        """
        api_key = self._get_api_key()
        if not api_key:
            provider_id = getattr(self, "__provider_id__", None)
            message = "Failed to authenticate with TypeSafe: no API key provided."
            if provider_id:
                message += f' Provide one in the provider config or the x-ogx-provider-data header, e.g. {{"{provider_id}_typesafe_api_key": "<API_KEY>"}} or {{"typesafe_api_key": "<API_KEY>"}}.'
            raise ValueError(message)

        return AsyncTypeSafeClient(
            api_key=api_key,
            base_url=self._get_base_url(),
            http_client=self._build_http_client(),
        )

    async def judge(self, request: SystemOneRequest) -> SystemOneResponse:
        """Answer a SystemOne judgment request by forwarding it to TypeSafe.

        ``request.model`` is already the provider-specific model id (the InferenceRouter rewrites
        it before dispatch). Questions are forwarded as raw dictionaries, which the SDK
        normalizes. The response is reconstructed into OGX's ``SystemOneResponse`` from the raw
        response body -- rather than cast from the SDK's parsed object -- so any drift between
        TypeSafe's schema and the OGX wire model is caught here and surfaces as a validation
        error instead of passing a wrong-shaped object through unchecked.
        """
        client = self._build_client()
        try:
            payload: dict[str, Any] = request.model_dump(exclude_none=True)
            sdk_response = await client.system_one(
                state=payload["state"],
                questions=payload["questions"],
                model=request.model,
            )
            return SystemOneResponse.model_validate_json(sdk_response.raw_http_response.content)
        except TypeSafeAPIError as exc:
            raise _to_provider_error(exc) from exc
        finally:
            await client.aclose()

    async def list_models(self) -> list[Model] | None:
        """List decision models from TypeSafe's ``GET /v1/models``.

        Each model is registered as ``ModelType.systemone`` with its description and release date
        preserved in metadata (the models router surfaces both in the TypeSafe-shaped listing).
        """
        api_key = self._get_api_key()
        if not api_key:
            logger.debug("list_models disabled because no TypeSafe API key is set")
            return None

        client = self._build_client()
        try:
            response = await client.models.list()
        except TypeSafeAPIError as exc:
            raise _to_provider_error(exc) from exc
        finally:
            await client.aclose()

        models: list[Model] = []
        for entry in response.models:
            if self.config.allowed_models is not None and entry.name not in self.config.allowed_models:
                continue
            models.append(
                Model(
                    identifier=entry.name,
                    provider_resource_id=entry.name,
                    provider_id=self.__provider_id__,
                    model_type=ModelType.systemone,
                    metadata={"description": entry.description, "release_date": entry.release_date},
                )
            )
        return models

    async def should_refresh_models(self) -> bool:
        return self.config.refresh_models

    async def register_model(self, model: Model) -> Model:
        return model

    async def unregister_model(self, model_id: str) -> None:
        return None

    ###
    # Operations this provider does not support (it serves judgment models only).
    #
    async def rerank(self, request: RerankRequest) -> RerankResponse:
        raise NotImplementedError(_UNIMPLEMENTED.format(operation="reranking"))

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        raise NotImplementedError(_UNIMPLEMENTED.format(operation="completions"))

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        raise NotImplementedError(_UNIMPLEMENTED.format(operation="chat completions"))

    async def openai_chat_completions_with_reasoning(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletionWithReasoning | AsyncIterator[Any]:
        raise NotImplementedError(_UNIMPLEMENTED.format(operation="reasoning chat completions"))

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        raise NotImplementedError(_UNIMPLEMENTED.format(operation="embeddings"))

    async def anthropic_messages(
        self,
        params: AnthropicCreateMessageRequest,
    ) -> AnthropicMessageResponse | AsyncIterator[AnthropicStreamEvent]:
        raise NotImplementedError(_UNIMPLEMENTED.format(operation="Anthropic messages"))

    async def anthropic_count_tokens(
        self,
        params: AnthropicCountTokensRequest,
    ) -> AnthropicCountTokensResponse:
        raise NotImplementedError(_UNIMPLEMENTED.format(operation="Anthropic token counting"))
