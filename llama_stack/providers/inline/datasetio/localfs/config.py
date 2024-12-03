# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from llama_stack.apis.datasetio import *  # noqa: F401, F403

from llama_stack.distribution.utils.config_dirs import RUNTIME_BASE_DIR
from llama_stack.providers.utils.kvstore.config import (
    KVStoreConfig,
    SqliteKVStoreConfig,
)


class LocalFSDatasetIOConfig(BaseModel):
    kvstore: KVStoreConfig = SqliteKVStoreConfig(
        db_path=(RUNTIME_BASE_DIR / "localfs_datasetio.db").as_posix()
    )  # Uses SQLite config specific to localfs storage
