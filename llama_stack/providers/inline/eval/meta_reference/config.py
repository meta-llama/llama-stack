# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from pydantic import BaseModel

from llama_stack.distribution.utils.config_dirs import RUNTIME_BASE_DIR
from llama_stack.providers.utils.kvstore.config import (
    KVStoreConfig,
    SqliteKVStoreConfig,
)


class MetaReferenceEvalConfig(BaseModel):
    kvstore: KVStoreConfig = SqliteKVStoreConfig(
        db_path=(RUNTIME_BASE_DIR / "meta_reference_eval.db").as_posix()
    )  # Uses SQLite config specific to Meta Reference Eval storage
