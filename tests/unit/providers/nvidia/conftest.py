# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

mock_session = MagicMock()
mock_session.closed = False
mock_session.close = AsyncMock()
mock_session.__aenter__ = AsyncMock(return_value=mock_session)
mock_session.__aexit__ = AsyncMock()


@pytest.fixture(scope="session", autouse=True)
def patch_aiohttp_session():
    with patch("aiohttp.ClientSession", return_value=mock_session):
        yield
