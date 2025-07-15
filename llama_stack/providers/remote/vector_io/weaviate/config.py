# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field


class WeaviateRequestProviderData(BaseModel):
    weaviate_api_key: str
    weaviate_cluster_url: str


class WeaviateVectorIOConfig(BaseModel):
    embedding_model: str | None = Field(
        default=None,
        description="Optional default embedding model for this provider. If not specified, will use system default.",
    )
    embedding_dimension: int | None = Field(
        default=None,
        description="Optional embedding dimension override. Only needed for models with variable dimensions (e.g., Matryoshka embeddings). If not specified, will auto-lookup from model registry.",
    )

    @classmethod
    def sample_run_config(cls, **kwargs: Any) -> dict[str, Any]:
        return {
            # Optional: Configure default embedding model for this provider
            # "embedding_model": "all-MiniLM-L6-v2",
            # "embedding_dimension": 384,  # Only needed for variable-dimension models
        }
