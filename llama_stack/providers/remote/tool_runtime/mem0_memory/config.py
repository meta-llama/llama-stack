# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from pydantic import BaseModel


class Mem0ToolRuntimeConfig(BaseModel):
    """Configuration for Mem0 Tool Runtime"""

    host: Optional[str] = "https://api.mem0.ai"
    api_key: Optional[str] = None
    top_k: int = 10
    org_id: Optional[str] = None
    project_id: Optional[str] = None
