# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import VectorIO

from .config import SampleVectorIOConfig


class SampleVectorIOImpl(VectorIO):
    def __init__(self, config: SampleVectorIOConfig):
        self.config = config

    async def register_vector_db(self, vector_db: VectorDB) -> None:
        # these are the vector dbs the Llama Stack will use to route requests to this provider
        # perform validation here if necessary
        pass

    async def initialize(self):
        pass

    async def shutdown(self):
        pass
