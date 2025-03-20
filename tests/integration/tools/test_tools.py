# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


def test_toolsgroups_unregister(llama_stack_client):
    client = llama_stack_client
    client.toolgroups.unregister(
        toolgroup_id="builtin::websearch",
    )
