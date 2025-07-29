# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations  # for forward references

import hashlib
import json
import os
import sqlite3
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
    normalized = {"method": method.upper(), "endpoint": parsed.path, "body": body}

    # Create hash - sort_keys=True ensures deterministic ordering
    normalized_json = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(normalized_json.encode()).hexdigest()


def get_inference_mode() -> str:
    return os.environ.get("LLAMA_STACK_INFERENCE_MODE", "live").lower()


def setup_inference_recording():
    mode = get_inference_mode()

    if mode not in ["live", "record", "replay"]:
        raise ValueError(f"Invalid LLAMA_STACK_INFERENCE_MODE: {mode}. Must be 'live', 'record', or 'replay'")

    if mode == "live":
        # Return a no-op context manager for live mode
        @contextmanager
        def live_mode():
            yield

        return live_mode()

    if "LLAMA_STACK_RECORDING_DIR" not in os.environ:
        raise ValueError("LLAMA_STACK_RECORDING_DIR must be set for recording or replaying")
    storage_dir = os.environ["LLAMA_STACK_RECORDING_DIR"]

    return inference_recording(mode=mode, storage_dir=storage_dir)


def _serialize_response(response: Any) -> Any:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    elif hasattr(response, "__dict__"):
        return dict(response.__dict__)
    else:
        return response


def _deserialize_response(data: dict[str, Any]) -> dict[str, Any]:
    return data


class ResponseStorage:
    """Handles SQLite index + JSON file storage/retrieval for inference recordings."""

    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
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
            f.write("\n")

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


async def _patched_inference_method(original_method, self, client_type, method_name=None, **kwargs):
    global _current_mode, _current_storage

    if _current_mode == "live" or _current_storage is None:
        # Normal operation
        return await original_method(self, **kwargs)

    # Get base URL and endpoint based on client type
    if client_type == "openai":
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

    elif client_type == "ollama":
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
    else:
        raise ValueError(f"Unknown client type: {client_type}")

    url = base_url.rstrip("/") + endpoint

    # Normalize request for matching
    method = "POST"
    headers = {}
    body = kwargs

    request_hash = normalize_request(method, url, headers, body)

    if _current_mode == "replay":
        recording = _current_storage.find_recording(request_hash)
        if recording:
            response_body = recording["response"]["body"]

            if recording["response"].get("is_streaming", False):

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
        response = await original_method(self, **kwargs)

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
    # Also import Ollama AsyncClient
    from ollama import AsyncClient as OllamaAsyncClient
    from openai.resources.chat.completions import AsyncCompletions as AsyncChatCompletions
    from openai.resources.completions import AsyncCompletions
    from openai.resources.embeddings import AsyncEmbeddings

    # Store original methods for both OpenAI and Ollama clients
    _original_methods = {
        "chat_completions_create": AsyncChatCompletions.create,
        "completions_create": AsyncCompletions.create,
        "embeddings_create": AsyncEmbeddings.create,
        "ollama_generate": OllamaAsyncClient.generate,
        "ollama_chat": OllamaAsyncClient.chat,
        "ollama_embed": OllamaAsyncClient.embed,
    }

    # Create patched methods for OpenAI client
    async def patched_chat_completions_create(self, **kwargs):
        return await _patched_inference_method(_original_methods["chat_completions_create"], self, "openai", **kwargs)

    async def patched_completions_create(self, **kwargs):
        return await _patched_inference_method(_original_methods["completions_create"], self, "openai", **kwargs)

    async def patched_embeddings_create(self, **kwargs):
        return await _patched_inference_method(_original_methods["embeddings_create"], self, "openai", **kwargs)

    # Apply OpenAI patches
    AsyncChatCompletions.create = patched_chat_completions_create
    AsyncCompletions.create = patched_completions_create
    AsyncEmbeddings.create = patched_embeddings_create

    # Create patched methods for Ollama client
    async def patched_ollama_generate(self, **kwargs):
        return await _patched_inference_method(
            _original_methods["ollama_generate"], self, "ollama", "generate", **kwargs
        )

    async def patched_ollama_chat(self, **kwargs):
        return await _patched_inference_method(_original_methods["ollama_chat"], self, "ollama", "chat", **kwargs)

    async def patched_ollama_embed(self, **kwargs):
        return await _patched_inference_method(_original_methods["ollama_embed"], self, "ollama", "embed", **kwargs)

    # Apply Ollama patches
    OllamaAsyncClient.generate = patched_ollama_generate
    OllamaAsyncClient.chat = patched_ollama_chat
    OllamaAsyncClient.embed = patched_ollama_embed


def unpatch_inference_clients():
    """Remove monkey patches and restore original OpenAI and Ollama client methods."""
    global _original_methods

    if not _original_methods:
        return

    # Import here to avoid circular imports
    from ollama import AsyncClient as OllamaAsyncClient
    from openai.resources.chat.completions import AsyncCompletions as AsyncChatCompletions
    from openai.resources.completions import AsyncCompletions
    from openai.resources.embeddings import AsyncEmbeddings

    # Restore OpenAI client methods
    AsyncChatCompletions.create = _original_methods["chat_completions_create"]
    AsyncCompletions.create = _original_methods["completions_create"]
    AsyncEmbeddings.create = _original_methods["embeddings_create"]

    # Restore Ollama client methods if they were patched
    OllamaAsyncClient.generate = _original_methods["ollama_generate"]
    OllamaAsyncClient.chat = _original_methods["ollama_chat"]
    OllamaAsyncClient.embed = _original_methods["ollama_embed"]

    _original_methods.clear()


@contextmanager
def inference_recording(mode: str = "live", storage_dir: str | Path | None = None) -> Generator[None, None, None]:
    """Context manager for inference recording/replaying."""
    global _current_mode, _current_storage

    # Set defaults
    if storage_dir is None:
        storage_dir_path = Path.home() / ".llama" / "recordings"
    else:
        storage_dir_path = Path(storage_dir)

    # Store previous state
    prev_mode = _current_mode
    prev_storage = _current_storage

    try:
        _current_mode = mode

        if mode in ["record", "replay"]:
            _current_storage = ResponseStorage(storage_dir_path)
            patch_inference_clients()

        yield

    finally:
        # Restore previous state
        if mode in ["record", "replay"]:
            unpatch_inference_clients()

        _current_mode = prev_mode
        _current_storage = prev_storage
