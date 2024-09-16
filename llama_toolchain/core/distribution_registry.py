# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from functools import lru_cache
from pathlib import Path
from typing import List, Optional
from .datatypes import *  # noqa: F403
import yaml


@lru_cache()
def available_distribution_specs() -> List[DistributionSpec]:
    distribution_specs = []
    for p in Path("llama_toolchain/configs/distributions/distribution_registry").rglob(
        "*.yaml"
    ):
        with open(p, "r") as f:
            distribution_specs.append(DistributionSpec(**yaml.safe_load(f)))

    return distribution_specs
