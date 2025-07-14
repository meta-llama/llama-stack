# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class OpenGaussVectorIOConfig(BaseModel):
    host: str | None = Field(default="localhost")
    port: int | None = Field(default=5432)
    db: str | None = Field(default="postgres")
    user: str | None = Field(default="postgres")
    password: str | None = Field(default="mysecretpassword")

    @classmethod
    def sample_run_config(
        cls,
        host: str = "${env.OPENGAUSS_HOST:=localhost}",
        port: str = "${env.OPENGAUSS_PORT:=5432}",
        db: str = "${env.OPENGAUSS_DB}",
        user: str = "${env.OPENGAUSS_USER}",
        password: str = "${env.OPENGAUSS_PASSWORD}",
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {"host": host, "port": port, "db": db, "user": user, "password": password}
