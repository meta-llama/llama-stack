# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict

from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Span


class SQLiteSpanProcessor(SpanProcessor):
    def __init__(self, conn_string, ttl_days=30):
        """Initialize the SQLite span processor with a connection string."""
        self.conn_string = conn_string
        self.ttl_days = ttl_days
        self._shutdown_event = threading.Event()
        self.cleanup_task = None
        self._thread_local = threading.local()
        self._connections: Dict[int, sqlite3.Connection] = {}
        self._lock = threading.Lock()
        self.setup_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-specific database connection."""
        thread_id = threading.get_ident()
        with self._lock:
            if thread_id not in self._connections:
                conn = sqlite3.connect(self.conn_string)
                self._connections[thread_id] = conn
            return self._connections[thread_id]

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

        # Start periodic cleanup in a separate thread
        self.cleanup_task = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self.cleanup_task.start()

    def _cleanup_old_data(self):
        """Delete records older than TTL."""
        try:
            conn = self._get_connection()
            cutoff_date = (datetime.now() - timedelta(days=self.ttl_days)).isoformat()
            cursor = conn.cursor()

            # Delete old span events
            cursor.execute(
                """
                DELETE FROM span_events
                WHERE span_id IN (
                    SELECT span_id FROM spans
                    WHERE trace_id IN (
                        SELECT trace_id FROM traces
                        WHERE created_at < ?
                    )
                )
            """,
                (cutoff_date,),
            )

            # Delete old spans
            cursor.execute(
                """
                DELETE FROM spans
                WHERE trace_id IN (
                    SELECT trace_id FROM traces
                    WHERE created_at < ?
                )
            """,
                (cutoff_date,),
            )

            # Delete old traces
            cursor.execute("DELETE FROM traces WHERE created_at < ?", (cutoff_date,))

            conn.commit()
            cursor.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def _periodic_cleanup(self):
        """Run cleanup periodically."""
        import time

        while not self._shutdown_event.is_set():
            time.sleep(3600)  # Sleep for 1 hour
            if not self._shutdown_event.is_set():
                self._cleanup_old_data()

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
        self._shutdown_event.set()

        # Wait for cleanup thread to finish if it exists
        if self.cleanup_task and self.cleanup_task.is_alive():
            self.cleanup_task.join(timeout=5.0)
        current_thread_id = threading.get_ident()

        with self._lock:
            # Close all connections from the current thread
            for thread_id, conn in list(self._connections.items()):
                if thread_id == current_thread_id:
                    try:
                        if conn:
                            conn.close()
                        del self._connections[thread_id]
                    except sqlite3.Error:
                        pass  # Ignore errors during shutdown

    def force_flush(self, timeout_millis=30000):
        """Force export of spans."""
        pass
