# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import Any, Dict


class RoutingTable:
    def __init__(self, provider_routing_table: Dict[str, Any]):
        self.provider_routing_table = provider_routing_table

    def print(self):
        print(f"ROUTING TABLE {self.provider_routing_table}")
