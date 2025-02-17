# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.telemetry import Telemetry

from .config import SampleConfig


class SampleTelemetryImpl(Telemetry):
    def __init__(self, config: SampleConfig):
        self.config = config

    async def initialize(self):
        pass
