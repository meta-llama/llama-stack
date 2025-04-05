# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

from pydantic import BaseModel

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol
from llama_stack.schema_utils import json_schema_type, webmethod


@json_schema_type
class Preprocessor(Resource):
    type: Literal[ResourceType.preprocessor.value] = ResourceType.preprocessor.value  # type: ignore

    @property
    def preprocessor_id(self) -> str:
        return self.identifier

    @property
    def provider_preprocessor_id(self) -> str:
        return self.provider_resource_id

    metadata: Optional[Dict[str, Any]] = None


class PreprocessorInput(BaseModel):
    preprocessor_id: str
    provider_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ListPreprocessorsResponse(BaseModel):
    data: List[Preprocessor]


@runtime_checkable
@trace_protocol
class Preprocessors(Protocol):
    @webmethod(route="/preprocessors", method="GET")
    async def list_preprocessors(self) -> ListPreprocessorsResponse: ...

    @webmethod(route="/preprocessors/{preprocessor_id:path}", method="GET")
    async def get_preprocessor(
        self,
        preprocessor_id: str,
    ) -> Preprocessor: ...

    @webmethod(route="/preprocessors", method="POST")
    async def register_preprocessor(
        self,
        preprocessor_id: str,
        provider_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Preprocessor: ...

    @webmethod(route="/preprocessors/{preprocessor_id:path}", method="DELETE")
    async def unregister_preprocessor(
        self,
        preprocessor_id: str,
    ) -> None: ...
