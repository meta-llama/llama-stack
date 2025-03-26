# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.datasets import Dataset, DatasetPurpose
from typing import Dict, List, Optional, Any

import logging

import psycopg_pool
from psycopg_pool.abc import ACT
from psycopg import sql
from urllib.parse import urlparse, parse_qs
from .config import PostgreSQLDatasetIOConfig
from typing import AsyncIterator

log = logging.getLogger(__name__)


class DatasetColumn:
    def __init__(self, name: str, is_array: bool):
        self.name = name
        self.is_array = is_array


class PgConnectionInfo:
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password

    def __str__(self):
        return f"host={self.host} port={self.port} dbname={self.database} user={self.user} password={self.password}"


def get_mandatory_cols(purpose: DatasetPurpose) -> list[DatasetColumn]:
    if purpose == DatasetPurpose.post_training_messages:
        return [DatasetColumn("messages", True)]
    elif purpose == DatasetPurpose.eval_question_answer:
        return [DatasetColumn("question", False), DatasetColumn("answer", False)]
    elif purpose == DatasetPurpose.eval_messages_answer:
        return [DatasetColumn("messages", True), DatasetColumn("answer", False)]
    else:
        raise ValueError(f"Unknown purpose: {purpose}")


def get_config_from_uri(uri: str, config: PostgreSQLDatasetIOConfig) -> tuple[PgConnectionInfo, str | None]:
    parsed = urlparse(uri)
    # Extract main components
    if parsed.scheme != "postgresql":
        raise ValueError(f"Unsupported scheme: {parsed.scheme} (uri: {uri})")

    # uri info has precedence over config info
    username = parsed.username if parsed.username else config.pg_user
    password = parsed.password if parsed.password else config.pg_password
    host = parsed.hostname if parsed.hostname else config.pg_host
    port = parsed.port if parsed.port else config.pg_port
    database = parsed.path.lstrip("/")  # Remove leading "/"
    database = database if database else config.pg_database

    # Extract query parameters
    raw_query = parsed.query.replace("?", "&")  # Fix multiple question marks
    query_params = parse_qs(raw_query)

    table = query_params.get("table", [None])[0]  # Extract first value if exists
    # TODO: read from metadata here?
    # if table is None:
    #     raise ValueError(f"Missing table parameter in URI: {uri}")

    return PgConnectionInfo(
        user=username,
        password=password,
        host=host,
        port=port,
        database=database,
    ), table


async def create_connection_pool(
    max_connections: int,
    info: PgConnectionInfo,
    min_connections: int = 1,
) -> psycopg_pool.AsyncConnectionPool:
    error = False
    try:
        pool = psycopg_pool.AsyncConnectionPool(
            str(info), min_size=min_connections, max_size=max_connections, open=False
        )
        await pool.open(wait=True, timeout=10.0)
    except Exception as e:
        log.error(f"Failed to create connection pool: {e}")
        error = True
        raise
    finally:
        if error and pool is not None:
            await pool.close()
            pool = None
    return pool


async def check_table_exists(conn: AsyncIterator[ACT], table_name: str) -> bool:
    try:
        sql_stmnt = "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)"
        cur = await conn.execute(sql_stmnt, (table_name,))
        row = await cur.fetchone()
        exists = bool(row[0])
        return exists
    except Exception as e:
        log.error(f"Error: {e}")
        raise
    finally:
        await cur.close()


async def _get_table_columns(conn: AsyncIterator[ACT], table_name: str) -> List[str]:
    try:
        query = sql.SQL("SELECT column_name FROM information_schema.columns WHERE table_name = {table}").format(
            table=table_name
        )
        async with conn.cursor() as cur:
            await cur.execute(query)
            table_cols = await cur.fetchall()
            return [col[0] for col in table_cols]
    except Exception as e:
        log.error(f"Error: {e}")
        raise


async def check_schema(conn: AsyncIterator[ACT], table_name: str, purpose: DatasetPurpose) -> None:
    try:
        # cur = await conn.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s",
        #                         (table_name,))
        table_cols = await _get_table_columns(conn, table_name)
        schema_cols = get_mandatory_cols(purpose)
        missing_col_names = []
        for sc in schema_cols:
            if not any(tc == sc.name for tc in table_cols):
                log.error(f"Failed to find column {sc.name} in table {table_name} (purpose {purpose})")
                missing_col_names.append(sc.name)
            else:
                # TODO: check type compatibility
                pass

        if len(missing_col_names) > 0:
            raise ValueError(f"Could not find column(s) {missing_col_names} in table {table_name} (purpose {purpose})")

    except Exception as e:
        log.error(f"Error: {e}")
        raise
    return


def get_table_name(dataset: Dataset) -> str:
    table_name = str(dataset.metadata.get("table", None))
    if table_name is None:
        log.error(f"No table defined for dataset: {dataset.provider_id}({dataset.identifier})")
        raise ValueError(f"No table defined for dataset: {dataset.identifier}")
    elif "'" in table_name or '"' in table_name:
        log.error(f"Table name {table_name} contains quotes - this is ignored for security reasons")
        raise ValueError(f"Table name {table_name} contains quotes - registration fails for security reasons")
    return table_name


async def check_table_and_schema(ds: Dataset, conn: AsyncIterator[ACT], provider_type: str) -> None:
    table_name = get_table_name(ds)
    # Check table existance
    try:
        exists = await check_table_exists(conn, table_name)
        if not exists:
            log.error(f'Table "{table_name}" does not exist')
            raise ValueError(
                f"Table '{table_name}' does not exist in the database, dataset '{ds.identifier}' cannot be registered"
            )
    except Exception as e:
        log.error(f"Error: {e}")
        raise

    # get and check table schema
    try:
        await check_schema(conn, table_name, ds.purpose)

    except Exception as e:
        log.error(f"Error: {e}")
        raise

    return


def build_select_statement(
    dataset: Dataset,
    conn: AsyncIterator[ACT],
    start_index: Optional[int] = None,
    limit: Optional[int] = None,
) -> str:
    """
    Build a select statement for the given purpose
    """
    params = []
    stmnt = "SELECT * from {} "
    params.append(sql.Identifier(dataset.metadata["table"]))

    if dataset.metadata.get("filter", None):
        stmnt += " WHERE {}"
        params.append(sql.Literal(dataset.metadata["filter"]))

    if limit is not None:
        stmnt += " LIMIT {}"
        params.append(sql.Literal(limit))

    if start_index is not None:
        stmnt += " OFFSET {}"
        params.append(sql.Literal(start_index))

    sql_stmnt = sql.SQL(stmnt).format(*params)

    return sql_stmnt


async def get_row_count(
    conn: AsyncIterator[ACT],
    table_name: str,
) -> int:
    """
    Get the number of rows in the table
    """
    try:
        sql_stmnt = "SELECT COUNT(*) FROM {}"
        sql_stmnt = sql.SQL(sql_stmnt).format(sql.Identifier(table_name))
        async with conn.cursor() as cur:
            await cur.execute(sql_stmnt)
            row_count = await cur.fetchone()
            return int(row_count[0])
    except Exception as e:
        log.error(f"Error: {e}")
        return 0


async def rows_to_iterrows_response(
    rows: List[Any],
    conn: AsyncIterator[ACT],
    table_name: str,
) -> List[Dict[str, Any]]:
    """
    Convert rows from the database to InterrowsResponse
    """
    res = []
    # cols = get_mandatory_cols(purpose)
    cols = await _get_table_columns(conn, table_name)
    for _i, row in enumerate(rows):
        res_row = {}
        for i, col in enumerate(cols):
            res_row[col] = row[i]
        res.append(res_row)
    return res
