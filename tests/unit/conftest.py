# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# We need to import the fixtures here so that pytest can find them
# but ruff doesn't think they are used and removes the import. "noqa: F401" prevents them from being removed
from .fixtures import cached_disk_dist_registry, disk_dist_registry, sqlite_kvstore  # noqa: F401
