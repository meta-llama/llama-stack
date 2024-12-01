# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from datetime import datetime
from typing import List, Optional

import psycopg2

from llama_stack.apis.telemetry import Span, SpanNode, Trace, TraceStore, TraceTree


class PostgresTraceStore(TraceStore):
    def __init__(self, conn_string: str):
        self.conn_string = conn_string

    async def get_trace(self, trace_id: str) -> Optional[TraceTree]:
        try:
            with psycopg2.connect(self.conn_string) as conn:
                with conn.cursor() as cur:
                    # Fetch all spans for the trace
                    cur.execute(
                        """
                        SELECT trace_id, span_id, parent_span_id, name,
                               start_time, end_time, attributes
                        FROM traces
                        WHERE trace_id = %s
                        """,
                        (trace_id,),
                    )
                    spans_data = cur.fetchall()

                    if not spans_data:
                        return None

                    # First pass: Build span map
                    span_map = {}
                    for span_data in spans_data:
                        # Ensure attributes is a string before parsing
                        attributes = span_data[6]
                        if isinstance(attributes, dict):
                            attributes = json.dumps(attributes)

                        span = Span(
                            span_id=span_data[1],
                            trace_id=span_data[0],
                            name=span_data[3],
                            start_time=span_data[4],
                            end_time=span_data[5],
                            parent_span_id=span_data[2],
                            attributes=json.loads(
                                attributes
                            ),  # Now safely parse the JSON string
                        )
                        span_map[span.span_id] = SpanNode(span=span)

                    # Second pass: Build parent-child relationships
                    root_node = None
                    for span_node in span_map.values():
                        parent_id = span_node.span.parent_span_id
                        if parent_id and parent_id in span_map:
                            span_map[parent_id].children.append(span_node)
                        elif not parent_id:
                            root_node = span_node

                    trace = Trace(
                        trace_id=trace_id,
                        root_span_id=root_node.span.span_id if root_node else "",
                        start_time=(
                            root_node.span.start_time if root_node else datetime.now()
                        ),
                        end_time=root_node.span.end_time if root_node else None,
                    )

                    return TraceTree(trace=trace, root=root_node)

        except Exception as e:
            raise Exception(
                f"Error querying PostgreSQL trace structure: {str(e)}"
            ) from e

    async def get_traces_for_sessions(self, session_ids: List[str]) -> List[Trace]:
        traces = []
        try:
            with psycopg2.connect(self.conn_string) as conn:
                with conn.cursor() as cur:
                    # Query traces for all session IDs
                    cur.execute(
                        """
                        SELECT DISTINCT trace_id, MIN(start_time) as start_time
                        FROM traces
                        WHERE attributes->>'session_id' = ANY(%s)
                        GROUP BY trace_id
                        """,
                        (session_ids,),
                    )
                    traces_data = cur.fetchall()

                    for trace_data in traces_data:
                        traces.append(
                            Trace(
                                trace_id=trace_data[0],
                                root_span_id="",
                                start_time=trace_data[1],
                            )
                        )

        except Exception as e:
            raise Exception(f"Error querying PostgreSQL traces: {str(e)}") from e

        return traces
