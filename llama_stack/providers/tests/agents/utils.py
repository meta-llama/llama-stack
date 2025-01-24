# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


async def create_agent_session(agents_impl, agent_config):
    create_response = await agents_impl.create_agent(agent_config)
    agent_id = create_response.agent_id

    # Create a session
    session_create_response = await agents_impl.create_agent_session(
        agent_id, "Test Session"
    )
    session_id = session_create_response.session_id
    return agent_id, session_id
