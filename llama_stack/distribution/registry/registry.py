# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import AbstractSet, Generic, TypeVar

TRegistry = TypeVar("TRegistry")


class Registry(Generic[TRegistry]):

    def __init__(self) -> None:
        super().__init__()
        self.registry = {}

    def names(self) -> AbstractSet[str]:
        return self.registry.keys()

    def register(self, name: str, task: TRegistry) -> None:
        if name in self.registry:
            raise ValueError(f"Dataset {name} already exists.")
        self.registry[name] = task

    def get(self, name: str) -> TRegistry:
        if name not in self.registry:
            raise ValueError(f"Dataset {name} not found.")
        return self.registry[name]

    def delete(self, name: str) -> None:
        if name not in self.registry:
            raise ValueError(f"Dataset {name} not found.")
        del self.registry[name]

    def reset(self) -> None:
        self.registry = {}
