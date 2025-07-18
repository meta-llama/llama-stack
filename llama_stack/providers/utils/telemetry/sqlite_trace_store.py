# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from datetime import datetime
from typing import Protocol

import aiosqlite

from llama_stack.apis.telemetry import QueryCondition, Span, SpanWithStatus, Trace


class TraceStore(Protocol):
    async def query_traces(
        self,
        attribute_filters: list[QueryCondition] | None = None,
        limit: int | None = 100,
        offset: int | None = 0,
        order_by: list[str] | None = None,
    ) -> list[Trace]: ...

    async def get_span_tree(
        self,
        span_id: str,
        attributes_to_return: list[str] | None = None,
        max_depth: int | None = None,
    ) -> dict[str, SpanWithStatus]: ...


class SQLiteTraceStore(TraceStore):
    def __init__(self, conn_string: str):
        self.conn_string = conn_string

    async def query_traces(
        self,
        attribute_filters: list[QueryCondition] | None = None,
        limit: int | None = 100,
        offset: int | None = 0,
        order_by: list[str] | None = None,
    ) -> list[Trace]:
        def build_where_clause() -> tuple[str, list]:
            if not attribute_filters:
                return "", []

            ops_map = {"eq": "=", "ne": "!=", "gt": ">", "lt": "<"}

            conditions = [
                f"json_extract(s.attributes, '$.{condition.key}') {ops_map[condition.op.value]} ?"
                for condition in attribute_filters
            ]
            params = [condition.value for condition in attribute_filters]
            where_clause = " WHERE " + " AND ".join(conditions)
            return where_clause, params

        def build_order_clause() -> str:
            if not order_by:
                return ""

            order_clauses = []
            for field in order_by:
                desc = field.startswith("-")
                clean_field = field[1:] if desc else field
                order_clauses.append(f"t.{clean_field} {'DESC' if desc else 'ASC'}")
            return " ORDER BY " + ", ".join(order_clauses)

        # Build the main query
        base_query = """
            WITH matching_traces AS (
                SELECT DISTINCT t.trace_id
                FROM traces t
                JOIN spans s ON t.trace_id = s.trace_id
                {where_clause}
            ),
            filtered_traces AS (
                SELECT t.trace_id, t.root_span_id, t.start_time, t.end_time
                FROM matching_traces mt
                JOIN traces t ON mt.trace_id = t.trace_id
                LEFT JOIN spans s ON t.trace_id = s.trace_id
                {order_clause}
            )
            SELECT DISTINCT trace_id, root_span_id, start_time, end_time
            FROM filtered_traces
            WHERE root_span_id IS NOT NULL
            LIMIT {limit} OFFSET {offset}
        """

        where_clause, params = build_where_clause()
        query = base_query.format(
            where_clause=where_clause,
            order_clause=build_order_clause(),
            limit=limit,
            offset=offset,
        )

        # Execute query and return results
        async with aiosqlite.connect(self.conn_string) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [
                    Trace(
                        trace_id=row["trace_id"],
                        root_span_id=row["root_span_id"],
                        start_time=datetime.fromisoformat(row["start_time"]),
                        end_time=datetime.fromisoformat(row["end_time"]),
                    )
                    for row in rows
                ]

    async def get_span_tree(
        self,
        span_id: str,
        attributes_to_return: list[str] | None = None,
        max_depth: int | None = None,
    ) -> dict[str, SpanWithStatus]:
        # Build the attributes selection
        attributes_select = "s.attributes"
        if attributes_to_return:
            json_object = ", ".join(f"'{key}', json_extract(s.attributes, '$.{key}')" for key in attributes_to_return)
            attributes_select = f"json_object({json_object})"

        # SQLite CTE query with filtered attributes
        query = f"""
        WITH RECURSIVE span_tree AS (
            SELECT s.*, 1 as depth, {attributes_select} as filtered_attributes
            FROM spans s
            WHERE s.span_id = ?

            UNION ALL

            SELECT s.*, st.depth + 1, {attributes_select} as filtered_attributes
            FROM spans s
            JOIN span_tree st ON s.parent_span_id = st.span_id
            WHERE (? IS NULL OR st.depth < ?)
        )
        SELECT *
        FROM span_tree
        ORDER BY depth, start_time
        """

        spans_by_id = {}
        async with aiosqlite.connect(self.conn_string) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(query, (span_id, max_depth, max_depth)) as cursor:
                rows = await cursor.fetchall()

                if not rows:
                    raise ValueError(f"Span {span_id} not found")

                for row in rows:
                    span = SpanWithStatus(
                        span_id=row["span_id"],
                        trace_id=row["trace_id"],
                        parent_span_id=row["parent_span_id"],
                        name=row["name"],
                        start_time=datetime.fromisoformat(row["start_time"]),
                        end_time=datetime.fromisoformat(row["end_time"]),
                        attributes=json.loads(row["filtered_attributes"]),
                        status=row["status"].lower(),
                    )

                    spans_by_id[span.span_id] = span

                return spans_by_id

    async def get_trace(self, trace_id: str) -> Trace:
        query = """
            SELECT *
            FROM traces t
            WHERE t.trace_id = ?
        """
        async with aiosqlite.connect(self.conn_string) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(query, (trace_id,)) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    raise ValueError(f"Trace {trace_id} not found")
                return Trace(**row)

    async def get_span(self, trace_id: str, span_id: str) -> Span:
        query = "SELECT * FROM spans WHERE trace_id = ? AND span_id = ?"
        async with aiosqlite.connect(self.conn_string) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(query, (trace_id, span_id)) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    raise ValueError(f"Span {span_id} not found")
                return Span(**row)
