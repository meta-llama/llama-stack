# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class EmbeddingConfig(BaseModel):
    """Configuration for embedding model used by vector-io providers.

    This allows providers to specify default embedding models for use-case specific
    vector stores, reducing the need for app developers to know embedding details.

    Example usage in provider config:
    ```yaml
    vector_io:
      - provider_id: question-answer
        provider_type: remote::pgvector
        config:
          embedding:
            model: prod/question-answer-embedder
            dimensions: 384
    ```
    """

    model: str = Field(description="The embedding model identifier to use")
    dimensions: int | None = Field(default=None, description="The embedding dimensions (optional, can be inferred)")

    def get_dimensions_or_default(self, default: int = 384) -> int:
        """Get dimensions with fallback to default if not specified."""
        return self.dimensions if self.dimensions is not None else default
