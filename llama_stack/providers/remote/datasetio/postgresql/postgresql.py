# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import logging
from typing import Any, Dict, List, Optional

from llama_stack.apis.datasetio import DatasetIO, IterrowsResponse
from llama_stack.apis.datasets import Dataset
from llama_stack.providers.datatypes import DatasetsProtocolPrivate
from llama_stack.providers.utils.common.provider_utils import get_provider_type
from llama_stack.providers.utils.kvstore import KVStore

# from llama_stack.providers.utils.datasetio.url_utils import get_dataframe_from_url
from llama_stack.providers.utils.kvstore import kvstore_impl

from psycopg_pool import AsyncConnectionPool

from .config import PostgreSQLDatasetIOConfig

#  from .pg_tools import get_config_from_uri, check_table_and_schema, create_connection_pool
from .pg_tools import get_config_from_uri, check_table_and_schema, create_connection_pool
from .pg_tools import build_select_statement, get_row_count, rows_to_iterrows_response
from .pg_tools import get_table_name

log = logging.getLogger(__name__)

DATASETS_PREFIX = "datasets:"


class PostgreSQLDatasetIOImpl(DatasetIO, DatasetsProtocolPrivate):
    def __init__(self, config: PostgreSQLDatasetIOConfig) -> None:
        self.config = config
        # local registry for keeping track of datasets within the provider
        self.dataset_infos: Dict[str, Dataset] = {}
        self.conn_pools: Dict[str, AsyncConnectionPool] = {}
        self.row_counts: Dict[str, int] = {}
        self.kvstore: KVStore | None = None

    async def initialize(self) -> None:
        self.kvstore = await kvstore_impl(self.config.kvstore)
        # Load existing datasets from kvstore
        start_key = DATASETS_PREFIX
        end_key = f"{DATASETS_PREFIX}\xff"
        stored_datasets = await self.kvstore.range(start_key, end_key)

        for ds in stored_datasets:
            dataset = Dataset.model_validate_json(ds)
            pg_config_info, _ = get_config_from_uri(dataset.source.uri, self.config)
            self.dataset_infos[dataset.identifier] = dataset
            try:
                conn_pool = await create_connection_pool(3, pg_config_info)
                self.conn_pools[dataset.identifier] = conn_pool
            except Exception as e:
                log.error(f"Failed to create connection pool for dataset on initialization {dataset.identifier}: {e}")

    async def shutdown(self) -> None: ...

    async def register_dataset(
        self,
        dataset_def: Dataset,
    ) -> None:
        # Store in kvstore
        provider_type = get_provider_type(self.__module__)
        if self.dataset_infos.get(dataset_def.identifier):
            log.error(
                f"Failed to register dataset {dataset_def.identifier}. " + "Dataset with this name alreadt exists"
            )
            raise ValueError(f"Dataset {dataset_def.identifier} already exists")

        pg_connection_info, table = get_config_from_uri(dataset_def.source.uri, self.config)
        tbmd = dataset_def.metadata.get("table", None)
        if table is not None:
            # Uri setting overrides metadata setting
            dataset_def.metadata["table"] = table  ## logging table for future use.

        if tbmd and table and tbmd != table:
            log.warning(
                f"Table name mismatch for dataset {dataset_def.identifier}: metadata:{tbmd} != uri:{table}, using {table}"
            )
        elif get_table_name(dataset_def) is None:  # should have been set by now
            log.error(
                f"No table defined for dataset: {provider_type}::{dataset_def.provider_id}({dataset_def.identifier})"
            )
            raise ValueError(f"No table defined for dataset {dataset_def.identifier}")

        try:
            pool = await create_connection_pool(3, pg_connection_info)
            async with pool.connection() as conn:
                await check_table_and_schema(dataset_def, conn, provider_type)
        except ValueError:
            # these are already logged in check_table_and_schema
            raise
        except Exception as e:
            log.error(f"Error: {e}")
            raise

        key = f"{DATASETS_PREFIX}{dataset_def.identifier}"

        await self.kvstore.set(
            key=key,
            value=dataset_def.model_dump_json(),
        )
        self.dataset_infos[dataset_def.identifier] = dataset_def
        self.conn_pools[dataset_def.identifier] = pool

    async def unregister_dataset(self, dataset_id: str) -> None:
        if self.conn_pools[dataset_id] is not None:
            await self.conn_pools[dataset_id].close()
            del self.conn_pools[dataset_id]
        key = f"{DATASETS_PREFIX}{dataset_id}"
        await self.kvstore.delete(key=key)
        del self.dataset_infos[dataset_id]

    async def iterrows(
        self,
        dataset_id: str,
        start_index: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> IterrowsResponse:
        if start_index is not None and start_index < 0:
            raise ValueError(f"start_index ({start_index}) must be a non-negative integer")

        dataset_def = self.dataset_infos[dataset_id]
        pool = self.conn_pools[dataset_id]
        if pool is None:  # Retry to crate connection pool
            try:
                pg_config_info, _ = get_config_from_uri(dataset_def.source.uri, self.config)
                pool = await create_connection_pool(3, pg_config_info)
                self.conn_pools[dataset_def.identifier] = pool
            except Exception as e:
                log.error(f"Failed to create connection pool for dataset {dataset_def.identifier}: {e}")
                raise

        try:
            async with pool.connection() as conn:
                await pool._check_connection(conn)
                stmnt = build_select_statement(dataset_def, conn, start_index=start_index, limit=limit)
                if self.row_counts.get(dataset_def.identifier) is None or start_index is None or start_index < 3:
                    # get row count only once per iteration
                    self.row_counts[dataset_def.identifier] = await get_row_count(conn, get_table_name(dataset_def))
                async with conn.cursor() as cur:
                    await cur.execute(stmnt)
                    rows = await cur.fetchall()
                    data = await rows_to_iterrows_response(rows, conn, get_table_name(dataset_def))
        except Exception as e:
            log.error(f"Error: {e}")
            raise

        begin = 0 if start_index is None else start_index
        end = begin + len(data)

        return IterrowsResponse(
            data=data,
            next_start_index=end if end < self.row_counts[dataset_def.identifier] else None,
        )

        # TODO: Implement filtering

    async def append_rows(self, dataset_id: str, rows: List[Dict[str, Any]]) -> None:
        # This inteface is not implemented in the DatasetsResource class and is not
        # accessible via the client.
        raise NotImplementedError("Uploading to postgresql dataset is not supported yet")
