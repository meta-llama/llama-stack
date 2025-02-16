# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class MongoDBVectorIOConfig(BaseModel):
    conncetion_str: str
    namespace: str = Field(None, description="Namespace of the MongoDB collection")
    index_name: Optional[str] = Field("default", description="Name of the index in the MongoDB collection") 
    filter_fields: Optional[str] = Field(None, description="Fields to filter the MongoDB collection")
    embedding_field: Optional[str] = Field("embeddings", description="Field name for the embeddings in the MongoDB collection")
    text_field: Optional[str] = Field("text", description="Field name for the text in the MongoDB collection")

    @classmethod
    def sample_config(cls) -> Dict[str, Any]:
        return {
            "connection_str": "{env.MONGODB_CONNECTION_STR}",
            "namespace": "{env.MONGODB_NAMESPACE}",
            "index_name": "{env.MONGODB_INDEX_NAME}",
            "filter_fields": "{env.MONGODB_FILTER_FIELDS}",
            "embedding_field": "{env.MONGODB_EMBEDDING_FIELD}",
            "text_field": "{env.MONGODB_TEXT_FIELD}",
        }
