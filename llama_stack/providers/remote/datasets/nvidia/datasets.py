# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from typing import Any, Dict, Literal, Optional

import aiohttp

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.common.type_system import ParamType
from llama_stack.apis.datasets.datasets import Dataset, ListDatasetsResponse
from llama_stack.apis.resource import ResourceType
from llama_stack.schema_utils import webmethod

from .config import NvidiaDatasetConfig


class NvidiaDatasetAdapter:
    """Nvidia NeMo Dataset API."""

    type: Literal[ResourceType.dataset.value] = ResourceType.dataset.value

    def __init__(self, config: NvidiaDatasetConfig):
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

    @webmethod(route="/datasets/{dataset_id:namespace}", method="GET")
    async def get_dataset(
        self,
        dataset_id: str,
    ) -> Optional[Dataset]:
        dataset_id, namespace = dataset_id.split(":")
        dataset = await self._make_request(
            method="GET",
            path=f"/v1/datasets/{namespace}/{dataset_id}",
        )
        created_at = datetime.fromisoformat(dataset.pop("created_at")) if "created_at" in dataset else datetime.now()
        identifier = dataset.pop("name")
        url = URL(uri=dataset.pop("files_url"))
        return Dataset(
            identifier=identifier,
            provider_id="nvidia",  # confirm this
            url=url,
            dataset_schema={},  # ToDo: get schema from the dataset
            created_at=created_at,
            metadata=dataset,
        )

    @webmethod(route="/datasets", method="GET")
    async def list_datasets(
        self,
    ) -> ListDatasetsResponse:
        ## ToDo: add pagination
        response = await self._make_request(method="GET", path="/v1/datasets")
        datasets = []
        for dataset in response["data"]:
            created_at = (
                datetime.fromisoformat(dataset.pop("created_at")) if "created_at" in dataset else datetime.now()
            )
            identifier = dataset.pop("name")
            url = URL(uri=dataset.pop("files_url"))
            datasets.append(
                Dataset(
                    identifier=identifier,
                    provider_id="nvidia",  # confirm this
                    url=url,
                    dataset_schema={},
                    created_at=created_at,
                    metadata=dataset,
                )
            )  # add remaining fields as metadata

        return ListDatasetsResponse(data=datasets)

    @webmethod(route="/datasets/{dataset_id:path}", method="POST")
    async def update_dataset(
        self,
        dataset_id: str,
        dataset_schema: Dict[str, ParamType],
        url: URL,
        provider_dataset_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    @webmethod(route="/datasets/{dataset_id:path}", method="DELETE")
    async def unregister_dataset(
        self,
        dataset_id: str,
        namespace: Optional[str] = "default",
    ) -> None: ...
