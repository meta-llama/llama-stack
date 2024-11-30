# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime, timedelta
from typing import List

import aiohttp

from llama_stack.apis.telemetry import Span, SpanNode, Trace, TraceStore, TraceTree


class JaegerTraceStore(TraceStore):
    def __init__(self, endpoint: str, service_name: str):
        self.endpoint = endpoint
        self.service_name = service_name

    async def get_trace(self, trace_id: str) -> TraceTree:
        params = {
            "traceID": trace_id,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.endpoint}/{trace_id}", params=params
                ) as response:
                    if response.status != 200:
                        raise Exception(
                            f"Failed to query Jaeger: {response.status} {await response.text()}"
                        )

                    trace_data = await response.json()
                    if not trace_data.get("data") or not trace_data["data"]:
                        return None

                    # First pass: Build span map
                    span_map = {}
                    for jaeger_span in trace_data["data"][0]["spans"]:
                        start_time = datetime.fromtimestamp(
                            jaeger_span["startTime"] / 1000000
                        )

                        # Some systems store end time directly in the span
                        if "endTime" in jaeger_span:
                            end_time = datetime.fromtimestamp(
                                jaeger_span["endTime"] / 1000000
                            )
                        else:
                            duration_microseconds = jaeger_span.get("duration", 0)
                            duration_timedelta = timedelta(
                                microseconds=duration_microseconds
                            )
                            end_time = start_time + duration_timedelta

                        span = Span(
                            span_id=jaeger_span["spanID"],
                            trace_id=trace_id,
                            name=jaeger_span["operationName"],
                            start_time=start_time,
                            end_time=end_time,
                            parent_span_id=next(
                                (
                                    ref["spanID"]
                                    for ref in jaeger_span.get("references", [])
                                    if ref["refType"] == "CHILD_OF"
                                ),
                                None,
                            ),
                            attributes={
                                tag["key"]: tag["value"]
                                for tag in jaeger_span.get("tags", [])
                            },
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
            raise Exception(f"Error querying Jaeger trace structure: {str(e)}") from e

    async def get_traces_for_sessions(self, session_ids: List[str]) -> List[Trace]:
        traces = []

        # Fetch traces for each session ID individually
        for session_id in session_ids:
            params = {
                "service": self.service_name,
                "tags": f'{{"session_id":"{session_id}"}}',
                "limit": 100,
                "lookback": "10000h",
            }

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.endpoint, params=params) as response:
                        if response.status != 200:
                            raise Exception(
                                f"Failed to query Jaeger: {response.status} {await response.text()}"
                            )

                        traces_data = await response.json()
                        seen_trace_ids = set()

                        for trace_data in traces_data.get("data", []):
                            trace_id = trace_data.get("traceID")
                            if trace_id and trace_id not in seen_trace_ids:
                                seen_trace_ids.add(trace_id)
                                traces.append(
                                    Trace(
                                        trace_id=trace_id,
                                        root_span_id="",
                                        start_time=datetime.now(),
                                    )
                                )

            except Exception as e:
                raise Exception(f"Error querying Jaeger traces: {str(e)}") from e

        return traces
