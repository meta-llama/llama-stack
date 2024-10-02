# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import SampleConfig


from llama_stack.apis.memory import *  # noqa: F403

from llama_stack.distribution.datatypes import RoutableProvider


class SampleMemoryImpl(Memory, RoutableProvider):
    def __init__(self, config: SampleConfig):
        self.config = config

    async def validate_routing_keys(self, routing_keys: list[str]) -> None:
        # these are the memory banks the Llama Stack will use to route requests to this provider
        # perform validation here if necessary
        pass

    async def initialize(self):
        pass
