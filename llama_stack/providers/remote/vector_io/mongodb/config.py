# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field


class MongoDBVectorIOConfig(BaseModel):
    connection_str: str = Field(None, description="Connection string for the MongoDB Atlas collection")
    namespace: str = Field(None, description="Namespace i.e. db_name.collection_name of the MongoDB Atlas collection")
    index_name: Optional[str] = Field("default", description="Name of the index in the MongoDB Atlas collection") 
    filter_fields: Optional[List[str]] = Field([], description="Fields to filter along side vector search in MongoDB Atlas collection")
    embeddings_key: Optional[str] = Field("embeddings", description="Field name for the embeddings in the MongoDB Atlas collection")
    text_field: Optional[str] = Field("text", description="Field name for the text in the MongoDB Atlas collection")


    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> Dict[str, Any]:
        return {
            "connection_str": "{env.MONGODB_CONNECTION_STR}",
            "namespace": "{env.MONGODB_NAMESPACE}",
        }