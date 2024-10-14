# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import AbstractSet, Dict, Generic, TypeVar

TRegistry = TypeVar("TRegistry")


class Registry(Generic[TRegistry]):
    _REGISTRY: Dict[str, TRegistry] = {}

    @staticmethod
    def names() -> AbstractSet[str]:
        return Registry._REGISTRY.keys()

    @staticmethod
    def register(name: str, task: TRegistry) -> None:
        if name in Registry._REGISTRY:
            raise ValueError(f"Dataset {name} already exists.")
        Registry._REGISTRY[name] = task

    @staticmethod
    def get(name: str) -> TRegistry:
        if name not in Registry._REGISTRY:
            raise ValueError(f"Dataset {name} not found.")
        return Registry._REGISTRY[name]

    @staticmethod
    def delete(name: str) -> None:
        if name not in Registry._REGISTRY:
            raise ValueError(f"Dataset {name} not found.")
        del Registry._REGISTRY[name]

    @staticmethod
    def reset() -> None:
        Registry._REGISTRY = {}
