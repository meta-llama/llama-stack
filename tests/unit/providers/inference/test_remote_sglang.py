# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ogx.providers.inline.responses.builtin.responses.types import AssistantMessageWithReasoning
from ogx.providers.remote.inference.sglang.config import SGLangInferenceAdapterConfig
from ogx.providers.remote.inference.sglang.sglang import SGLangInferenceAdapter
from ogx_api import (
    HealthStatus,
    ModelType,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIUserMessageParam,
)
from ogx_api.inference import RerankRequest

# Unit tests for the remote SGLang provider. They cover what is specific to
# SGLang's HTTP surface (health probe, rerank wire format, reasoning history);
# API-level behaviour is covered by tests/integration/inference/.
#
# pytest tests/unit/providers/inference/test_remote_sglang.py -v


def _adapter(**config) -> SGLangInferenceAdapter:
    config.setdefault("base_url", "http://mocked.localhost:30000/v1")
    adapter = SGLangInferenceAdapter(config=SGLangInferenceAdapterConfig(**config))
    adapter.__provider_id__ = "sglang"
    return adapter


def _mock_client(mock_client_class, response):
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    client.post = AsyncMock(return_value=response)
    mock_client_class.return_value.__aenter__.return_value = client
    return client


async def test_initialize_requires_base_url():
    adapter = SGLangInferenceAdapter(config=SGLangInferenceAdapterConfig())
    with pytest.raises(ValueError, match="SGLANG_URL"):
        await adapter.initialize()


@pytest.mark.parametrize(
    "base_url, expected",
    [
        ("http://localhost:30000/v1", "http://localhost:30000"),
        ("http://localhost:30000/v1/", "http://localhost:30000"),
        ("http://localhost:30000", "http://localhost:30000"),
        ("http://localhost:30000/", "http://localhost:30000"),
    ],
)
def test_server_root_strips_version(base_url, expected):
    assert _adapter(base_url=base_url)._get_base_url_without_version() == expected


async def test_health_probes_server_root():
    adapter = _adapter(base_url="http://localhost:30000/v1/")
    with patch("httpx2.AsyncClient") as mock_client_class:
        response = MagicMock()
        response.raise_for_status.return_value = None
        client = _mock_client(mock_client_class, response)

        health = await adapter.health()

    assert health["status"] == HealthStatus.OK
    client.get.assert_called_once_with("http://localhost:30000/health")


async def test_health_reports_connection_errors():
    adapter = _adapter()
    with patch("httpx2.AsyncClient") as mock_client_class:
        client = MagicMock()
        client.get.side_effect = Exception("Connection failed")
        mock_client_class.return_value.__aenter__.return_value = client

        health = await adapter.health()

    assert health["status"] == HealthStatus.ERROR
    assert "Health check failed: Connection failed" in health["message"]


def test_models_are_classified_with_models_dev():
    adapter = _adapter()
    assert adapter.construct_model_from_identifier("text-embedding-3-large").model_type == ModelType.embedding
    assert adapter.construct_model_from_identifier("text-embedding-3-large").provider_id == "sglang"
    assert adapter.construct_model_from_identifier("zai-org/GLM-5.3-Flash").model_type == ModelType.llm


async def test_chat_completion_applies_configured_max_tokens():
    adapter = _adapter(max_tokens=512)
    captured = {}

    async def fake_super(params):
        captured["params"] = params
        return MagicMock()

    with patch.object(
        SGLangInferenceAdapter.__mro__[2], "openai_chat_completion", new=AsyncMock(side_effect=fake_super)
    ):
        await adapter.openai_chat_completion(
            OpenAIChatCompletionRequestWithExtraBody(
                model="m", messages=[OpenAIUserMessageParam(role="user", content="hi")]
            )
        )

    assert captured["params"].max_tokens == 512


async def test_reasoning_history_is_sent_as_reasoning_content():
    """SGLang reads prior-turn thinking from ``reasoning_content``, so the key must survive."""
    adapter = _adapter()
    captured = {}

    async def fake_chat_completion(params):
        captured["messages"] = params.messages

        async def chunks():
            if False:
                yield None

        return chunks()

    adapter.openai_chat_completion = fake_chat_completion
    params = OpenAIChatCompletionRequestWithExtraBody(
        model="m",
        stream=True,
        messages=[
            OpenAIUserMessageParam(role="user", content="hi"),
            AssistantMessageWithReasoning(role="assistant", content="ok", reasoning_content="thinking"),
        ],
    )
    stream = await adapter.openai_chat_completions_with_reasoning(params)
    assert [chunk async for chunk in stream] == []

    assistant = captured["messages"][1]
    assert assistant["reasoning_content"] == "thinking"
    assert "reasoning" not in assistant


async def test_non_streaming_reasoning_is_not_supported():
    adapter = _adapter()
    params = OpenAIChatCompletionRequestWithExtraBody(
        model="m", messages=[OpenAIUserMessageParam(role="user", content="hi")]
    )
    with pytest.raises(NotImplementedError):
        await adapter.openai_chat_completions_with_reasoning(params)


async def test_rerank_uses_sglang_wire_format():
    adapter = _adapter(base_url="https://sglang.example.com/v1", api_token="my-secret-token")
    adapter.get_request_provider_data = MagicMock(return_value=None)

    with patch("httpx2.AsyncClient") as mock_client_class:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = [
            {"score": 0.1, "index": 0, "meta_info": {}},
            {"score": 0.9, "index": 1, "document": "doc2"},
        ]
        client = _mock_client(mock_client_class, response)

        result = await adapter.rerank(
            RerankRequest(model="rerank-model", query="test", items=["doc1", "doc2"], max_num_results=2)
        )

    call = client.post.call_args
    assert call.args[0] == "https://sglang.example.com/v1/rerank"
    assert call.kwargs["json"] == {
        "query": "test",
        "documents": ["doc1", "doc2"],
        "top_n": 2,
    }
    assert call.kwargs["headers"]["Authorization"] == "Bearer my-secret-token"
    assert [(d.index, d.relevance_score) for d in result.data] == [(1, 0.9), (0, 0.1)]


async def test_rerank_without_token_sends_no_auth_header():
    adapter = _adapter()
    adapter.get_request_provider_data = MagicMock(return_value=None)

    with patch("httpx2.AsyncClient") as mock_client_class:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = [{"score": 0.5, "index": 0}]
        client = _mock_client(mock_client_class, response)

        await adapter.rerank(RerankRequest(model="rerank-model", query="test", items=["doc1"]))

    assert "Authorization" not in client.post.call_args.kwargs["headers"]


@pytest.mark.parametrize("body", [{"results": []}, [{"index": 0}]])
async def test_rerank_rejects_unexpected_responses(body):
    adapter = _adapter()
    adapter.get_request_provider_data = MagicMock(return_value=None)

    with patch("httpx2.AsyncClient") as mock_client_class:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = body
        _mock_client(mock_client_class, response)

        with pytest.raises(RuntimeError):
            await adapter.rerank(RerankRequest(model="rerank-model", query="test", items=["doc1"]))


async def test_rerank_passes_tls_config():
    adapter = _adapter(base_url="https://sglang.example.com/v1", network={"tls": {"verify": False}})
    adapter.get_request_provider_data = MagicMock(return_value=None)

    with patch("httpx2.AsyncClient") as mock_client_class:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = [{"score": 0.5, "index": 0}]
        _mock_client(mock_client_class, response)

        await adapter.rerank(RerankRequest(model="rerank-model", query="test", items=["doc1"]))

    assert mock_client_class.call_args.kwargs.get("verify") is False
