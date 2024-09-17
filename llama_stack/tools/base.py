# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import List

from llama_stack.apis.inference import Message


class BaseTool(ABC):
    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def run(self, messages: List[Message]) -> List[Message]:
        raise NotImplementedError
