# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from pydantic import BaseModel


class LlamaStackDistribution(BaseModel):
    name: str
    description: str

    # you must install the packages to get the functionality needed.
    # later, we may have a docker image be the main artifact of
    # a distribution.
    pip_packages: List[str]
