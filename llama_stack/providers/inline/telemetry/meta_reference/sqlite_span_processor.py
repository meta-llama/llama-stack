# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os
import sqlite3
from datetime import datetime

from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Span


class SQLiteSpanProcessor(SpanProcessor):
    def __init__(self, conn_string):
        """Initialize the SQLite span processor with a connection string."""
        self.conn_string = conn_string
        self.conn = None
        self.setup_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get the database connection."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.conn_string, check_same_thread=False)
        return self.conn

    def setup_database(self):
        """Create the necessary tables if they don't exist."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.conn_string), exist_ok=True)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT PRIMARY KEY,
                service_name TEXT,
                root_span_id TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS spans (
                span_id TEXT PRIMARY KEY,
                trace_id TEXT REFERENCES traces(trace_id),
                parent_span_id TEXT,
                name TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                attributes TEXT,
                status TEXT,
                kind TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS span_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                span_id TEXT REFERENCES spans(span_id),
                name TEXT,
                timestamp TIMESTAMP,
                attributes TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_traces_created_at
            ON traces(created_at)
        """
        )

        conn.commit()
        cursor.close()

    def on_start(self, span: Span, parent_context=None):
        """Called when a span starts."""
        pass

    def on_end(self, span: Span):
        """Called when a span ends. Export the span data to SQLite."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            trace_id = format(span.get_span_context().trace_id, "032x")
            span_id = format(span.get_span_context().span_id, "016x")
            service_name = span.resource.attributes.get("service.name", "unknown")

            parent_span_id = None
            parent_context = span.parent
            if parent_context:
                parent_span_id = format(parent_context.span_id, "016x")

            # Insert into traces
            cursor.execute(
                """
                INSERT INTO traces (
                    trace_id, service_name, root_span_id, start_time, end_time
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(trace_id) DO UPDATE SET
                    root_span_id = COALESCE(root_span_id, excluded.root_span_id),
                    start_time = MIN(excluded.start_time, start_time),
                    end_time = MAX(excluded.end_time, end_time)
            """,
                (
                    trace_id,
                    service_name,
                    (span_id if not parent_span_id else None),
                    datetime.fromtimestamp(span.start_time / 1e9).isoformat(),
                    datetime.fromtimestamp(span.end_time / 1e9).isoformat(),
                ),
            )

            # Insert into spans
            cursor.execute(
                """
                INSERT INTO spans (
                    span_id, trace_id, parent_span_id, name,
                    start_time, end_time, attributes, status,
                    kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    span_id,
                    trace_id,
                    parent_span_id,
                    span.name,
                    datetime.fromtimestamp(span.start_time / 1e9).isoformat(),
                    datetime.fromtimestamp(span.end_time / 1e9).isoformat(),
                    json.dumps(dict(span.attributes)),
                    span.status.status_code.name,
                    span.kind.name,
                ),
            )

            for event in span.events:
                cursor.execute(
                    """
                    INSERT INTO span_events (
                        span_id, name, timestamp, attributes
                    ) VALUES (?, ?, ?, ?)
                """,
                    (
                        span_id,
                        event.name,
                        datetime.fromtimestamp(event.timestamp / 1e9).isoformat(),
                        json.dumps(dict(event.attributes)),
                    ),
                )

            conn.commit()
            cursor.close()
        except Exception as e:
            print(f"Error exporting span to SQLite: {e}")

    def shutdown(self):
        """Cleanup any resources."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def force_flush(self, timeout_millis=30000):
        """Force export of spans."""
        pass