# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from typing import List, Optional

from prometheus_api_client import PrometheusConnect

from llama_stack.apis.telemetry import (
    GetMetricsResponse,
    MetricDataPoint,
    MetricLabelMatcher,
    MetricQueryType,
    MetricSeries,
)

from .metrics_store import MetricsStore


class PrometheusMetricsStore(MetricsStore):
    def __init__(self, endpoint: str, disable_ssl: bool = True):
        self.prom = PrometheusConnect(url=endpoint, disable_ssl=disable_ssl)

    async def get_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        step: Optional[str] = "15s",
        query_type: MetricQueryType = MetricQueryType.RANGE,
        label_matchers: Optional[List[MetricLabelMatcher]] = None,
    ) -> GetMetricsResponse:
        try:
            query = metric_name
            if label_matchers:
                matchers = [f'{m.name}{m.operator.value}"{m.value}"' for m in label_matchers]
                query = f"{metric_name}{{{','.join(matchers)}}}"

            if query_type == MetricQueryType.INSTANT:
                result = self.prom.custom_query(query=query)
                result = [{"metric": r["metric"], "values": [[r["value"][0], r["value"][1]]]} for r in result]
            else:
                result = self.prom.custom_query_range(
                    query=query,
                    start_time=start_time,
                    end_time=end_time if end_time else None,
                    step=step,
                )

            series = []
            for metric_data in result:
                values = [
                    MetricDataPoint(timestamp=datetime.fromtimestamp(point[0]), value=float(point[1]))
                    for point in metric_data["values"]
                ]
                series.append(MetricSeries(metric=metric_name, labels=metric_data.get("metric", {}), values=values))

            return GetMetricsResponse(data=series)

        except Exception as e:
            print(f"Error querying metrics: {e}")
            return GetMetricsResponse(data=[])
