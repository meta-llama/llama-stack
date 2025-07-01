# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import hashlib
import uuid


def generate_chunk_id(document_id: str, chunk_text: str) -> str:
    """
    Generate a unique chunk ID using a hash of the document ID and chunk text.

    Note: MD5 is used only to calculate an identifier, not for security purposes.
    Adding usedforsecurity=False for compatibility with FIPS environments.
    """
    hash_input = f"{document_id}:{chunk_text}".encode()

    try:
        md5_hash = hashlib.md5(hash_input, usedforsecurity=False).hexdigest()
    except TypeError:
        # Fallback for environments that don't support usedforsecurity (e.g., Python < 3.9 or non-OpenSSL backends)
        md5_hash = hashlib.md5(hash_input).hexdigest()

    return str(uuid.UUID(md5_hash))
