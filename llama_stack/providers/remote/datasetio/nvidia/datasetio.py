# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Optional

import aiohttp

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.common.type_system import ParamType
from llama_stack.apis.datasetio import IterrowsResponse
from llama_stack.schema_utils import webmethod

from .config import NvidiaDatasetIOConfig


class NvidiaDatasetIOAdapter:
    """Nvidia NeMo DatasetIO API."""

    def __init__(self, config: NvidiaDatasetIOConfig):
        self.config = config
        self.headers = {}
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key}"

    async def _make_request(
        self,
        method: str,
        path: str,
        headers: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Helper method to make HTTP requests to the Customizer API."""
        url = f"{self.config.datasets_url}{path}"
        request_headers = self.headers.copy()  # Create a copy to avoid modifying the original

        if headers:
            request_headers.update(headers)

        # Add content-type header for JSON requests
        if json and "Content-Type" not in request_headers:
            request_headers["Content-Type"] = "application/json"

        async with aiohttp.ClientSession(headers=request_headers) as session:
            async with session.request(method, url, params=params, json=json, **kwargs) as response:
                if response.status >= 400:
                    error_data = await response.json()
                    raise Exception(f"API request failed: {error_data}")
                return await response.json()

    @webmethod(route="/datasets", method="POST")
    async def register_dataset(
        self,
        dataset_id: str,
        dataset_schema: Dict[str, ParamType],
        url: URL,
        provider_dataset_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a new dataset.

        Args:
            dataset_id: The ID of the dataset.
            dataset_schema: The schema of the dataset.
            url: The URL of the dataset.
            provider_dataset_id: The ID of the provider dataset.
            provider_id: The ID of the provider.
            metadata: The metadata of the dataset.

        Returns:
            None
        """
        ...

    @webmethod(route="/datasets/{dataset_id:path}", method="POST")
    async def update_dataset(
        self,
        dataset_id: str,
        dataset_schema: Dict[str, ParamType],
        url: URL,
        provider_dataset_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError("Not implemented")

    @webmethod(route="/datasets/{dataset_id:path}", method="DELETE")
    async def unregister_dataset(
        self,
        dataset_id: str,
        namespace: Optional[str] = "default",
    ) -> None:
        raise NotImplementedError("Not implemented")

    async def iterrows(
        self,
        dataset_id: str,
        start_index: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> IterrowsResponse:
        raise NotImplementedError("Not implemented")

    async def append_rows(self, dataset_id: str, rows: List[Dict[str, Any]]) -> None:
        raise NotImplementedError("Not implemented")
