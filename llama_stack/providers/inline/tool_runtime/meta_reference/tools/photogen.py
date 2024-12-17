# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict

from llama_stack.providers.inline.tool_runtime.meta_reference.tools.base import BaseTool
from pydantic import BaseModel


class PhotogenConfig(BaseModel):
    dump_dir: str


class PhotogenTool(BaseTool):

    @classmethod
    def tool_id(cls) -> str:
        return "photogen"

    @classmethod
    def get_provider_config_type(cls):
        return PhotogenConfig

    async def execute(self, query: str) -> Dict:
        config = PhotogenConfig(**self.config)
        """
        Implement this to give the model an ability to generate images.

        Return:
            info = {
                "filepath": str(image_filepath),
                "mimetype": "image/png",
            }
        """
        raise NotImplementedError()
