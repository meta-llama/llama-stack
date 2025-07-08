# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import hashlib
import uuid


def generate_chunk_id(document_id: str, chunk_text: str, chunk_window: str | None = None) -> str:
    """
    Generate a unique chunk ID using a hash of the document ID and chunk text.

    Note: MD5 is used only to calculate an identifier, not for security purposes.
    Adding usedforsecurity=False for compatibility with FIPS environments.
    """
    hash_input = f"{document_id}:{chunk_text}".encode()
    if chunk_window:
        hash_input += f":{chunk_window}".encode()
    return str(uuid.UUID(hashlib.md5(hash_input, usedforsecurity=False).hexdigest()))
