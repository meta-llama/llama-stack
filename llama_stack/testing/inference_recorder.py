# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

# Global state for the recording system
_current_mode: str | None = None
_current_storage: ResponseStorage | None = None
_original_methods: dict[str, Any] = {}


def normalize_request(method: str, url: str, headers: dict[str, Any], body: dict[str, Any]) -> str:
    """Create a normalized hash of the request for consistent matching."""
    # Extract just the endpoint path
    from urllib.parse import urlparse

    parsed = urlparse(url)
    endpoint = parsed.path

    # Create normalized request dict
    normalized: dict[str, Any] = {
        "method": method.upper(),
        "endpoint": endpoint,
    }

    # Normalize body parameters
    if body:
        # Handle model parameter
        if "model" in body:
            normalized["model"] = body["model"]

        # Handle messages (normalize whitespace)
        if "messages" in body:
            normalized_messages = []
            for msg in body["messages"]:
                normalized_msg = dict(msg)
                if "content" in normalized_msg and isinstance(normalized_msg["content"], str):
                    # Normalize whitespace
                    normalized_msg["content"] = " ".join(normalized_msg["content"].split())
                normalized_messages.append(normalized_msg)
            normalized["messages"] = normalized_messages

        # Handle other parameters (sort for consistency)
        other_params = {}
        for key, value in body.items():
            if key not in ["model", "messages"]:
                if isinstance(value, float):
                    # Round floats to 6 decimal places
                    other_params[key] = round(value, 6)
                else:
                    other_params[key] = value

        if other_params:
            # Sort dictionary keys for consistent hashing
            normalized["parameters"] = dict(sorted(other_params.items()))

    # Create hash
    normalized_json = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(normalized_json.encode()).hexdigest()


def get_current_test_id() -> str:
    """Extract test ID from pytest context or fall back to environment/generated ID."""
    # Try to get from pytest context
    try:
        import _pytest.fixtures

        if hasattr(_pytest.fixtures, "_current_request") and _pytest.fixtures._current_request:
            request = _pytest.fixtures._current_request
            if hasattr(request, "node"):
                # Use the test node ID as our test identifier
                node_id: str = request.node.nodeid
                # Clean up the node ID to be filesystem-safe
                test_id = node_id.replace("/", "_").replace("::", "_").replace(".py", "")
                return test_id
    except AttributeError:
        pass

    # Fall back to environment-based or generated ID
    return os.environ.get("LLAMA_STACK_TEST_ID", f"test_{uuid.uuid4().hex[:8]}")


def get_inference_mode() -> str:
    """Get the inference recording mode from environment variables."""
    return os.environ.get("LLAMA_STACK_INFERENCE_MODE", "live").lower()


def setup_inference_recording():
    """Convenience function to set up inference recording based on environment variables."""
    mode = get_inference_mode()

    if mode not in ["live", "record", "replay"]:
        raise ValueError(f"Invalid LLAMA_STACK_INFERENCE_MODE: {mode}. Must be 'live', 'record', or 'replay'")

    if mode == "live":
        # Return a no-op context manager for live mode
        @contextmanager
        def live_mode():
            yield

        return live_mode()

    test_id = get_current_test_id()
    storage_dir = os.environ.get("LLAMA_STACK_RECORDING_DIR", str(Path.home() / ".llama" / "recordings"))

    return inference_recording(mode=mode, test_id=test_id, storage_dir=storage_dir)


def _serialize_response(response: Any) -> Any:
    """Serialize OpenAI response objects to JSON-compatible format."""
    if hasattr(response, "model_dump"):
        return response.model_dump()
    elif hasattr(response, "__dict__"):
        return dict(response.__dict__)
    else:
        return response


def _deserialize_response(data: dict[str, Any]) -> dict[str, Any]:
    """Deserialize response data back to a dict format."""
    # For simplicity, just return the dict - this preserves all the data
    # The original response structure is sufficient for replaying
    return data


class ResponseStorage:
    """Handles SQLite index + JSON file storage/retrieval for inference recordings."""

    def __init__(self, base_dir: Path, test_id: str):
        self.base_dir = base_dir
        self.test_id = test_id
        self.test_dir = base_dir / test_id
        self.responses_dir = self.test_dir / "responses"
        self.db_path = self.test_dir / "index.sqlite"

        self._ensure_directories()
        self._init_database()

    def _ensure_directories(self):
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(exist_ok=True)

    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recordings (
                    request_hash TEXT PRIMARY KEY,
                    response_file TEXT,
                    endpoint TEXT,
                    model TEXT,
                    timestamp TEXT,
                    is_streaming BOOLEAN
                )
            """)

    def store_recording(self, request_hash: str, request: dict[str, Any], response: dict[str, Any]):
        """Store a request/response pair."""
        # Generate unique response filename
        response_file = f"{request_hash[:12]}.json"
        response_path = self.responses_dir / response_file

        # Serialize response body if needed
        serialized_response = dict(response)
        if "body" in serialized_response:
            if isinstance(serialized_response["body"], list):
                # Handle streaming responses (list of chunks)
                serialized_response["body"] = [_serialize_response(chunk) for chunk in serialized_response["body"]]
            else:
                # Handle single response
                serialized_response["body"] = _serialize_response(serialized_response["body"])

        # Save response to JSON file
        with open(response_path, "w") as f:
            json.dump({"request": request, "response": serialized_response}, f, indent=2)

        # Update SQLite index
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO recordings
                (request_hash, response_file, endpoint, model, timestamp, is_streaming)
                VALUES (?, ?, ?, ?, datetime('now'), ?)
            """,
                (
                    request_hash,
                    response_file,
                    request.get("endpoint", ""),
                    request.get("model", ""),
                    response.get("is_streaming", False),
                ),
            )

    def find_recording(self, request_hash: str) -> dict[str, Any] | None:
        """Find a recorded response by request hash."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT response_file FROM recordings WHERE request_hash = ?", (request_hash,)
            ).fetchone()

        if not result:
            return None

        response_file = result[0]
        response_path = self.responses_dir / response_file

        if not response_path.exists():
            return None

        with open(response_path) as f:
            data = json.load(f)

        # Deserialize response body if needed
        if "response" in data and "body" in data["response"]:
            if isinstance(data["response"]["body"], list):
                # Handle streaming responses
                data["response"]["body"] = [_deserialize_response(chunk) for chunk in data["response"]["body"]]
            else:
                # Handle single response
                data["response"]["body"] = _deserialize_response(data["response"]["body"])

        return cast(dict[str, Any], data)


async def _patched_create_method(original_method, self, **kwargs):
    """Patched version of OpenAI client create methods."""
    global _current_mode, _current_storage

    if _current_mode == "live" or _current_storage is None:
        # Normal operation
        return await original_method(self, **kwargs)

    # Get base URL from the client
    base_url = str(self._client.base_url)

    # Determine endpoint based on the method's module/class path
    method_str = str(original_method)
    if "chat.completions" in method_str:
        endpoint = "/v1/chat/completions"
    elif "embeddings" in method_str:
        endpoint = "/v1/embeddings"
    elif "completions" in method_str:
        endpoint = "/v1/completions"
    else:
        # Fallback - try to guess from the self object
        if hasattr(self, "_resource") and hasattr(self._resource, "_resource"):
            resource_name = getattr(self._resource._resource, "_resource", "unknown")
            if "chat" in str(resource_name):
                endpoint = "/v1/chat/completions"
            elif "embeddings" in str(resource_name):
                endpoint = "/v1/embeddings"
            else:
                endpoint = "/v1/completions"
        else:
            endpoint = "/v1/completions"

    url = base_url.rstrip("/") + endpoint

    # Normalize request for matching
    method = "POST"
    headers = {}
    body = kwargs

    request_hash = normalize_request(method, url, headers, body)

    if _current_mode == "replay":
        # Try to find recorded response
        recording = _current_storage.find_recording(request_hash)
        if recording:
            # Return recorded response
            response_body = recording["response"]["body"]

            # Handle streaming responses
            if recording["response"].get("is_streaming", False):
                # For streaming, we need to return an async iterator
                async def replay_stream():
                    for chunk in response_body:
                        yield chunk

                return replay_stream()
            else:
                return response_body
        else:
            raise RuntimeError(
                f"No recorded response found for request hash: {request_hash}\n"
                f"Endpoint: {endpoint}\n"
                f"Model: {body.get('model', 'unknown')}\n"
                f"To record this response, run with LLAMA_STACK_INFERENCE_MODE=record"
            )

    elif _current_mode == "record":
        # Make real request and record it
        response = await original_method(self, **kwargs)

        # Store the recording
        request_data = {
            "method": method,
            "url": url,
            "headers": headers,
            "body": body,
            "endpoint": endpoint,
            "model": body.get("model", ""),
        }

        # Determine if this is a streaming request based on request parameters
        is_streaming = body.get("stream", False)

        if is_streaming:
            # For streaming responses, we need to collect all chunks immediately before yielding
            # This ensures the recording is saved even if the generator isn't fully consumed
            chunks = []
            async for chunk in response:
                chunks.append(chunk)

            # Store the recording immediately
            response_data = {"body": chunks, "is_streaming": True}
            _current_storage.store_recording(request_hash, request_data, response_data)

            # Return a generator that replays the stored chunks
            async def replay_recorded_stream():
                for chunk in chunks:
                    yield chunk

            return replay_recorded_stream()
        else:
            response_data = {"body": response, "is_streaming": False}
            _current_storage.store_recording(request_hash, request_data, response_data)
            return response

    else:
        return await original_method(self, **kwargs)


async def _patched_ollama_method(original_method, self, method_name, **kwargs):
    """Patched version of Ollama AsyncClient methods."""
    global _current_mode, _current_storage

    if _current_mode == "live" or _current_storage is None:
        # Normal operation
        return await original_method(self, **kwargs)

    # Get base URL from the client (Ollama client uses host attribute)
    base_url = getattr(self, "host", "http://localhost:11434")
    if not base_url.startswith("http"):
        base_url = f"http://{base_url}"

    # Determine endpoint based on method name
    if method_name == "generate":
        endpoint = "/api/generate"
    elif method_name == "chat":
        endpoint = "/api/chat"
    elif method_name == "embed":
        endpoint = "/api/embeddings"
    else:
        endpoint = f"/api/{method_name}"

    url = base_url.rstrip("/") + endpoint

    # Normalize request for matching
    method = "POST"
    headers = {}
    body = kwargs

    request_hash = normalize_request(method, url, headers, body)

    if _current_mode == "replay":
        # Try to find recorded response
        recording = _current_storage.find_recording(request_hash)
        if recording:
            # Return recorded response
            response_body = recording["response"]["body"]

            # Handle streaming responses for Ollama
            if recording["response"].get("is_streaming", False):
                # For streaming, we need to return an async iterator
                async def replay_ollama_stream():
                    for chunk in response_body:
                        yield chunk

                return replay_ollama_stream()
            else:
                return response_body
        else:
            raise RuntimeError(
                f"No recorded response found for request hash: {request_hash}\n"
                f"Endpoint: {endpoint}\n"
                f"Model: {body.get('model', 'unknown')}\n"
                f"To record this response, run with LLAMA_STACK_INFERENCE_MODE=record"
            )

    elif _current_mode == "record":
        # Make real request and record it
        response = await original_method(self, **kwargs)

        # Store the recording
        request_data = {
            "method": method,
            "url": url,
            "headers": headers,
            "body": body,
            "endpoint": endpoint,
            "model": body.get("model", ""),
        }

        # Determine if this is a streaming request based on request parameters
        is_streaming = body.get("stream", False)

        if is_streaming:
            # For streaming responses, we need to collect all chunks immediately before yielding
            # This ensures the recording is saved even if the generator isn't fully consumed
            chunks = []
            async for chunk in response:
                chunks.append(chunk)

            # Store the recording immediately
            response_data = {"body": chunks, "is_streaming": True}
            _current_storage.store_recording(request_hash, request_data, response_data)

            # Return a generator that replays the stored chunks
            async def replay_recorded_stream():
                for chunk in chunks:
                    yield chunk

            return replay_recorded_stream()
        else:
            response_data = {"body": response, "is_streaming": False}
            _current_storage.store_recording(request_hash, request_data, response_data)
            return response

    else:
        return await original_method(self, **kwargs)


def patch_inference_clients():
    """Install monkey patches for OpenAI client methods and Ollama AsyncClient methods."""
    global _original_methods

    # Import here to avoid circular imports
    from openai import AsyncOpenAI
    from openai.resources.chat.completions import AsyncCompletions as AsyncChatCompletions
    from openai.resources.completions import AsyncCompletions
    from openai.resources.embeddings import AsyncEmbeddings

    # Also import Ollama AsyncClient
    try:
        from ollama import AsyncClient as OllamaAsyncClient
    except ImportError:
        ollama_async_client = None
    else:
        ollama_async_client = OllamaAsyncClient

    # Store original methods for both OpenAI and Ollama clients
    _original_methods = {
        "chat_completions_create": AsyncChatCompletions.create,
        "completions_create": AsyncCompletions.create,
        "embeddings_create": AsyncEmbeddings.create,
    }

    # Add Ollama client methods if available
    if ollama_async_client:
        _original_methods.update(
            {
                "ollama_generate": ollama_async_client.generate,
                "ollama_chat": ollama_async_client.chat,
                "ollama_embed": ollama_async_client.embed,
            }
        )

    # Create patched methods for OpenAI client
    async def patched_chat_completions_create(self, **kwargs):
        return await _patched_create_method(_original_methods["chat_completions_create"], self, **kwargs)

    async def patched_completions_create(self, **kwargs):
        return await _patched_create_method(_original_methods["completions_create"], self, **kwargs)

    async def patched_embeddings_create(self, **kwargs):
        return await _patched_create_method(_original_methods["embeddings_create"], self, **kwargs)

    # Apply OpenAI patches
    AsyncChatCompletions.create = patched_chat_completions_create
    AsyncCompletions.create = patched_completions_create
    AsyncEmbeddings.create = patched_embeddings_create

    # Create patched methods for Ollama client
    if ollama_async_client:

        async def patched_ollama_generate(self, **kwargs):
            return await _patched_ollama_method(_original_methods["ollama_generate"], self, "generate", **kwargs)

        async def patched_ollama_chat(self, **kwargs):
            return await _patched_ollama_method(_original_methods["ollama_chat"], self, "chat", **kwargs)

        async def patched_ollama_embed(self, **kwargs):
            return await _patched_ollama_method(_original_methods["ollama_embed"], self, "embed", **kwargs)

        # Apply Ollama patches
        ollama_async_client.generate = patched_ollama_generate
        ollama_async_client.chat = patched_ollama_chat
        ollama_async_client.embed = patched_ollama_embed

    # Also try to patch the AsyncOpenAI __init__ to trace client creation
    original_openai_init = AsyncOpenAI.__init__

    def patched_openai_init(self, *args, **kwargs):
        result = original_openai_init(self, *args, **kwargs)

        # After client is created, try to re-patch its methods
        if hasattr(self, "chat") and hasattr(self.chat, "completions"):
            original_chat_create = self.chat.completions.create

            async def instance_patched_chat_create(**kwargs):
                return await _patched_create_method(original_chat_create, self.chat.completions, **kwargs)

            self.chat.completions.create = instance_patched_chat_create

        return result

    AsyncOpenAI.__init__ = patched_openai_init


def unpatch_inference_clients():
    """Remove monkey patches and restore original OpenAI and Ollama client methods."""
    global _original_methods

    if not _original_methods:
        return

    # Import here to avoid circular imports
    from openai.resources.chat.completions import AsyncCompletions as AsyncChatCompletions
    from openai.resources.completions import AsyncCompletions
    from openai.resources.embeddings import AsyncEmbeddings

    # Restore OpenAI client methods
    if "chat_completions_create" in _original_methods:
        AsyncChatCompletions.create = _original_methods["chat_completions_create"]

    if "completions_create" in _original_methods:
        AsyncCompletions.create = _original_methods["completions_create"]

    if "embeddings_create" in _original_methods:
        AsyncEmbeddings.create = _original_methods["embeddings_create"]

    # Restore Ollama client methods if they were patched
    try:
        from ollama import AsyncClient as OllamaAsyncClient

        if "ollama_generate" in _original_methods:
            OllamaAsyncClient.generate = _original_methods["ollama_generate"]

        if "ollama_chat" in _original_methods:
            OllamaAsyncClient.chat = _original_methods["ollama_chat"]

        if "ollama_embed" in _original_methods:
            OllamaAsyncClient.embed = _original_methods["ollama_embed"]

    except ImportError:
        pass

    _original_methods.clear()


@contextmanager
def inference_recording(
    mode: str = "live", test_id: str | None = None, storage_dir: str | Path | None = None
) -> Generator[None, None, None]:
    """Context manager for inference recording/replaying."""
    global _current_mode, _current_storage

    # Set defaults
    if storage_dir is None:
        storage_dir_path = Path.home() / ".llama" / "recordings"
    else:
        storage_dir_path = Path(storage_dir)

    if test_id is None:
        test_id = f"test_{uuid.uuid4().hex[:8]}"

    # Store previous state
    prev_mode = _current_mode
    prev_storage = _current_storage

    try:
        _current_mode = mode

        if mode in ["record", "replay"]:
            _current_storage = ResponseStorage(storage_dir_path, test_id)
            patch_inference_clients()

        yield

    finally:
        # Restore previous state
        if mode in ["record", "replay"]:
            unpatch_inference_clients()

        _current_mode = prev_mode
        _current_storage = prev_storage
