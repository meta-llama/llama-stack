# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

"""Global vector-store configuration shared across the stack.

This module introduces `VectorStoreConfig`, a small Pydantic model that
lives under `StackRunConfig.vector_store_config`.  It lets deployers set
an explicit default embedding model (and dimension) that the Vector-IO
router will inject whenever the caller does not specify one.
"""

import os

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["VectorStoreConfig"]


class VectorStoreConfig(BaseModel):
    """Stack-level defaults for vector-store creation.

    Attributes
    ----------
    default_embedding_model
        The model *id* the stack should use when an embedding model is
        required but not supplied by the API caller.  When *None* the
        router will raise a :class:`~llama_stack.errors.MissingEmbeddingModelError`.
    default_embedding_dimension
        Optional integer hint for vector dimension.  Routers/providers
        may validate that the chosen model emits vectors of this size.
    """

    default_embedding_model: str | None = Field(
        default_factory=lambda: os.getenv("LLAMA_STACK_DEFAULT_EMBEDDING_MODEL")
    )
    default_embedding_dimension: int | None = Field(
        default_factory=lambda: int(os.getenv("LLAMA_STACK_DEFAULT_EMBEDDING_DIMENSION", 0)) or None, ge=1
    )

    model_config = ConfigDict(frozen=True)
