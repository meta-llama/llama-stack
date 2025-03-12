# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, IAny, nc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
from typing import Tuple

from .config import NvidiaPostTrainingConfig

logger = logging.getLogger(__name__)


# ToDo: implement post health checks for customizer are enabled
async def _get_health(url: str) -> Tuple[bool, bool]: ...


async def check_health(config: NvidiaPostTrainingConfig) -> None: ...
