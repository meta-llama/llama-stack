# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from llama_stack.apis.telemetry import (
    GetMetricsResponse,
    MetricLabelMatcher,
    MetricQueryType,
)


class MetricsStore(ABC):
    @abstractmethod
    async def get_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        step: Optional[str] = "15s",
        query_type: MetricQueryType = MetricQueryType.RANGE,
        label_matchers: Optional[List[MetricLabelMatcher]] = None,
    ) -> GetMetricsResponse:
        pass
