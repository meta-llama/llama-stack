# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from datetime import datetime

import psycopg2
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Span


class PostgresSpanProcessor(SpanProcessor):
    def __init__(self, conn_string):
        """Initialize the PostgreSQL span processor with a connection string."""
        self.conn_string = conn_string
        self.conn = None
        self.setup_database()

    def setup_database(self):
        """Create the necessary table if it doesn't exist."""
        with psycopg2.connect(self.conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS traces (
                        trace_id TEXT,
                        span_id TEXT,
                        parent_span_id TEXT,
                        name TEXT,
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        attributes JSONB,
                        status TEXT,
                        kind TEXT,
                        service_name TEXT,
                        session_id TEXT
                    )
                """
                )
            conn.commit()

    def on_start(self, span: Span, parent_context=None):
        """Called when a span starts."""
        pass

    def on_end(self, span: Span):
        """Called when a span ends. Export the span data to PostgreSQL."""
        try:
            with psycopg2.connect(self.conn_string) as conn:
                with conn.cursor() as cur:

                    cur.execute(
                        """
                        INSERT INTO traces (
                            trace_id, span_id, parent_span_id, name,
                            start_time, end_time, attributes, status,
                            kind, service_name, session_id
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                        (
                            format(span.get_span_context().trace_id, "032x"),
                            format(span.get_span_context().span_id, "016x"),
                            (
                                format(span.parent.span_id, "016x")
                                if span.parent
                                else None
                            ),
                            span.name,
                            datetime.fromtimestamp(span.start_time / 1e9),
                            datetime.fromtimestamp(span.end_time / 1e9),
                            json.dumps(dict(span.attributes)),
                            span.status.status_code.name,
                            span.kind.name,
                            span.resource.attributes.get("service.name", "unknown"),
                            span.attributes.get("session_id", None),
                        ),
                    )
                conn.commit()
        except Exception as e:
            print(f"Error exporting span to PostgreSQL: {e}")

    def shutdown(self):
        """Cleanup any resources."""
        if self.conn:
            self.conn.close()

    def force_flush(self, timeout_millis=30000):
        """Force export of spans."""
        pass
