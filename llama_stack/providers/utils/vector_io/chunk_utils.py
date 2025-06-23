# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import hashlib
import logging
import uuid

from llama_stack.apis.vector_io import Chunk


def generate_chunk_id(document_id: str, chunk_text: str) -> str:
    """Generate a unique chunk ID using a hash of document ID and chunk text."""
    hash_input = f"{document_id}:{chunk_text}".encode()
    return str(uuid.UUID(hashlib.md5(hash_input).hexdigest()))


def extract_chunk_id_from_metadata(chunk: Chunk) -> str | None:
    """Extract existing chunk ID from metadata. This is for compatibility with older Chunks
    that stored the document_id in the metadata and not in the ChunkMetadata."""
    if chunk.chunk_metadata is not None and hasattr(chunk.chunk_metadata, "chunk_id"):
        return chunk.chunk_metadata.chunk_id

    if "chunk_id" in chunk.metadata:
        return str(chunk.metadata["chunk_id"])

    return None


def extract_or_generate_chunk_id(chunk: Chunk) -> str:
    """Extract existing chunk ID or generate a new one if not present. This is for compatibility with older Chunks
    that stored the document_id in the metadata."""
    stored_chunk_id = extract_chunk_id_from_metadata(chunk)
    if stored_chunk_id:
        return stored_chunk_id
    elif "document_id" in chunk.metadata:
        return generate_chunk_id(chunk.metadata["document_id"], str(chunk.content))
    else:
        logging.warning("Chunk has no ID or document_id in metadata. Generating random ID.")
        return str(uuid.uuid4())
