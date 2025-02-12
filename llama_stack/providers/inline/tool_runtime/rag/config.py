# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class DoclingConfig(BaseModel):
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    export_type: Optional[str] = Field(
        default="MARKDOWN",
        description="The export type, one of MARKDOWN (default) or JSON",
    )
    accelerator_options: Optional[dict[str, any]] = Field(
        default=None,
        description="Optional key value arguments passed to the accelerator options",
    )


class RagToolRuntimeConfig(BaseModel):
    docling: Optional[DoclingConfig] = Field(
        default_factory=DoclingConfig,
        description="The optional Docling settings",
    )
    chunker: Optional[str] = Field(
        default=None,
        description="The chunker implementation, one of default or docling",
    )

    @classmethod
    def sample_config(cls) -> dict[str, any]:
        return {"docling": {"export_type": "MARKDOWN"}, "chunker": "default"}
