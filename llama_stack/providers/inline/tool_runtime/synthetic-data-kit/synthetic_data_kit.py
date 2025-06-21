# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import asyncio
import logging
import mimetypes
import os
import tempfile
from typing import Any

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.files import Files
from llama_stack.apis.tools import (
    ListToolDefsResponse,
    ToolDef,
    ToolGroup,
    ToolInvocationResult,
    ToolParameter,
    ToolRuntime,
)
from llama_stack.providers.datatypes import ToolGroupsProtocolPrivate
from llama_stack.providers.utils.memory.vector_store import content_from_data_and_mime_type

from .config import SyntheticDataKitToolRuntimeConfig

log = logging.getLogger(__name__)


class SyntheticDataKitToolRuntimeImpl(ToolGroupsProtocolPrivate, ToolRuntime):
    def __init__(
        self,
        config: SyntheticDataKitToolRuntimeConfig,
        files_api: Files,
    ):
        self.config = config
        self.files_api = files_api

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def register_toolgroup(self, toolgroup: ToolGroup) -> None:
        pass

    async def unregister_toolgroup(self, toolgroup_id: str) -> None:
        return

    async def list_runtime_tools(
        self, tool_group_id: str | None = None, mcp_endpoint: URL | None = None
    ) -> ListToolDefsResponse:
        return ListToolDefsResponse(
            data=[
                ToolDef(
                    name="convert_file_to_text",
                    description="Convert a file to text",
                    parameters=[
                        ToolParameter(
                            name="file_id",
                            description="The id of the file to convert.",
                            parameter_type="string",
                        ),
                    ],
                ),
            ]
        )

    async def invoke_tool(self, tool_name: str, kwargs: dict[str, Any]) -> ToolInvocationResult:
        if tool_name != "convert_file_to_text":
            raise ValueError(f"Unknown tool: {tool_name}")

        file_id = kwargs["file_id"]
        file_response = await self.files_api.openai_retrieve_file(file_id)
        mime_type = self._guess_mime_type(file_response.filename)
        content_response = await self.files_api.openai_retrieve_file_content(file_id)

        mime_category = mime_type.split("/")[0] if mime_type else None
        if mime_category == "text":
            # Don't use synthetic-data-kit if the file is already text
            content = content_from_data_and_mime_type(content_response.body, mime_type)
            return ToolInvocationResult(
                content=content,
                metadata={},
            )
        else:
            return await asyncio.to_thread(
                self._synthetic_data_kit_convert, content_response.body, file_response.filename
            )

    def _guess_mime_type(self, filename: str) -> str | None:
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type is None and filename.endswith(".md"):
            mime_type = "text/markdown"
        return mime_type

    def _synthetic_data_kit_convert(self, content_body: bytes, filename: str) -> ToolInvocationResult:
        from synthetic_data_kit.core.ingest import process_file

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, filename)
                with open(file_path, "wb") as f:
                    f.write(content_body)
                output_path = process_file(file_path, tmpdir)
                with open(output_path) as f:
                    content = f.read()

                    return ToolInvocationResult(
                        content=content,
                        metadata={},
                    )
        except Exception as e:
            return ToolInvocationResult(
                content="",
                error_message=f"Error converting file: {e}",
                error_code=1,
                metadata={},
            )
