# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from pydantic import BaseModel


class SQLiteVectorIOConfig(BaseModel):
    db_path: str

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> Dict[str, Any]:
        return {
            "db_path": "${env.SQLITE_STORE_DIR:" + __distro_dir__ + "}/" + "sqlite_vec.db",
        }
