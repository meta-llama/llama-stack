# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Optional

from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.telemetry import QueryCondition, Span, SpanWithChildren


class TelemetryDatasetMixin:
    """Mixin class that provides dataset-related functionality for telemetry providers."""

    datasetio_api: DatasetIO

    async def save_spans_to_dataset(
        self,
        attribute_filters: List[QueryCondition],
        attributes_to_save: List[str],
        dataset_id: str,
        max_depth: Optional[int] = None,
    ) -> None:
        spans = await self.query_spans(
            attribute_filters=attribute_filters,
            attributes_to_return=attributes_to_save,
            max_depth=max_depth,
        )

        rows = [
            {
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "parent_span_id": span.parent_span_id,
                "name": span.name,
                "start_time": span.start_time,
                "end_time": span.end_time,
                **{attr: span.attributes.get(attr) for attr in attributes_to_save},
            }
            for span in spans
        ]

        await self.datasetio_api.append_rows(dataset_id=dataset_id, rows=rows)

    async def query_spans(
        self,
        attribute_filters: List[QueryCondition],
        attributes_to_return: List[str],
        max_depth: Optional[int] = None,
    ) -> List[Span]:
        traces = await self.query_traces(attribute_filters=attribute_filters)
        spans = []

        for trace in traces:
            span_tree = await self.get_span_tree(
                span_id=trace.root_span_id,
                attributes_to_return=attributes_to_return,
                max_depth=max_depth,
            )

            def extract_spans(span: SpanWithChildren) -> List[Span]:
                result = []
                if span.attributes and all(
                    attr in span.attributes and span.attributes[attr] is not None
                    for attr in attributes_to_return
                ):
                    result.append(
                        Span(
                            trace_id=trace.root_span_id,
                            span_id=span.span_id,
                            parent_span_id=span.parent_span_id,
                            name=span.name,
                            start_time=span.start_time,
                            end_time=span.end_time,
                            attributes=span.attributes,
                        )
                    )

                for child in span.children:
                    result.extend(extract_spans(child))

                return result

            spans.extend(extract_spans(span_tree))

        return spans
