# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

def pytest_collection_modifyitems(items):
    for item in items:
        item.name = item.name.replace(' ', '_') 
