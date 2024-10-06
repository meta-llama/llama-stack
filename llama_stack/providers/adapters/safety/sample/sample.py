# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import SampleConfig


from llama_stack.apis.safety import *  # noqa: F403

from llama_stack.distribution.datatypes import RoutableProvider


class SampleSafetyImpl(Safety, RoutableProvider):
    def __init__(self, config: SampleConfig):
        self.config = config

    async def validate_routing_keys(self, routing_keys: list[str]) -> None:
        # these are the safety shields the Llama Stack will use to route requests to this provider
        # perform validation here if necessary
        pass

    async def initialize(self):
        pass
