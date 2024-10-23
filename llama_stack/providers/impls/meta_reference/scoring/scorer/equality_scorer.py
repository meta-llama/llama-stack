# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.impls.meta_reference.scoring.scorer.base_scorer import (
    BaseScorer,
)


class EqualityScorer(BaseScorer):
    """
    A scorer that assigns a score of 1.0 if the input string matches the target string, and 0.0 otherwise.
    """

    def __init__(self, target: str) -> None:
        """
        Initialize the EqualityScorer with a target string.

        Args:
            target (str): The target string to match against.
        """
        self.target = target
