# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import Any

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class QdrantVectorIOConfig(BaseModel):
    path: str
    embedding_model: str | None = Field(
        default=None,
        description="Optional default embedding model for this provider. If not specified, will use system default.",
    )
    embedding_dimension: int | None = Field(
        default=None,
        description="Optional embedding dimension override. Only needed for models with variable dimensions (e.g., Matryoshka embeddings). If not specified, will auto-lookup from model registry.",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "path": "${env.QDRANT_PATH:=~/.llama/" + __distro_dir__ + "}/" + "qdrant.db",
            # Optional: Configure default embedding model for this provider
            # "embedding_model": "all-MiniLM-L6-v2",
            # "embedding_dimension": 384,  # Only needed for variable-dimension models
        }
