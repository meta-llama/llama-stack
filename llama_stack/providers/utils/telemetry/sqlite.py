# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from datetime import datetime
from typing import List, Optional

import aiosqlite

from llama_stack.apis.telemetry import (
    MaterializedSpan,
    QueryCondition,
    Trace,
    TraceStore,
)


class SQLiteTraceStore(TraceStore):
    def __init__(self, conn_string: str):
        self.conn_string = conn_string

    async def query_traces(
        self,
        attribute_conditions: Optional[List[QueryCondition]] = None,
        attribute_keys_to_return: Optional[List[str]] = None,
        limit: Optional[int] = 100,
        offset: Optional[int] = 0,
        order_by: Optional[List[str]] = None,
    ) -> List[Trace]:
        # Build the SQL query with attribute selection
        select_clause = """
            SELECT DISTINCT t.trace_id, t.root_span_id, t.start_time, t.end_time
        """
        if attribute_keys_to_return:
            for key in attribute_keys_to_return:
                select_clause += (
                    f", json_extract(s.attributes, '$.{key}') as attr_{key}"
                )

        query = (
            select_clause
            + """
            FROM traces t
            JOIN spans s ON t.trace_id = s.trace_id
        """
        )
        params = []

        # Add attribute conditions if present
        if attribute_conditions:
            conditions = []
            for condition in attribute_conditions:
                conditions.append(
                    f"json_extract(s.attributes, '$.{condition.key}') {condition.op} ?"
                )
                params.append(condition.value)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        # Add ordering
        if order_by:
            order_clauses = []
            for field in order_by:
                desc = False
                if field.startswith("-"):
                    field = field[1:]
                    desc = True
                order_clauses.append(f"t.{field} {'DESC' if desc else 'ASC'}")
            query += " ORDER BY " + ", ".join(order_clauses)

        # Add limit and offset
        query += f" LIMIT {limit} OFFSET {offset}"

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

    async def get_materialized_span(
        self,
        span_id: str,
        attribute_keys_to_return: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
    ) -> MaterializedSpan:
        # Build the attributes selection
        attributes_select = "s.attributes"
        if attribute_keys_to_return:
            json_object = ", ".join(
                f"'{key}', json_extract(s.attributes, '$.{key}')"
                for key in attribute_keys_to_return
            )
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

        async with aiosqlite.connect(self.conn_string) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(query, (span_id, max_depth, max_depth)) as cursor:
                rows = await cursor.fetchall()

                if not rows:
                    raise ValueError(f"Span {span_id} not found")

                # Build span tree
                spans_by_id = {}
                root_span = None

                for row in rows:
                    span = MaterializedSpan(
                        span_id=row["span_id"],
                        trace_id=row["trace_id"],
                        parent_span_id=row["parent_span_id"],
                        name=row["name"],
                        start_time=datetime.fromisoformat(row["start_time"]),
                        end_time=datetime.fromisoformat(row["end_time"]),
                        attributes=json.loads(row["filtered_attributes"]),
                        status=row["status"].lower(),
                        children=[],
                    )

                    spans_by_id[span.span_id] = span

                    if span.span_id == span_id:
                        root_span = span
                    elif span.parent_span_id in spans_by_id:
                        spans_by_id[span.parent_span_id].children.append(span)

                return root_span
