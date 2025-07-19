# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

from llama_stack.distribution.access_control.conditions import (
    UserWithValueInList,
    UserWithValueNotInList,
    parse_condition,
)
from llama_stack.distribution.request_headers import user_from_scope

from .middleware_base import BaseServerMiddleware


class AccessControlMiddleware(BaseServerMiddleware):
    async def process_request(self, scope, receive, send, _route, _impl, webmethod):
        if webmethod and webmethod.access_control:
            # Extract user from scope (set by auth middleware)
            # Note: We can't use get_authenticated_user() here because the middleware
            # runs before the request_provider_data_context is established
            user = user_from_scope(scope)

            if not _evaluate_policy(webmethod.access_control, user):
                # Send 403 Forbidden response
                await send(
                    {
                        "type": "http.response.start",
                        "status": 403,
                        "headers": [[b"content-type", b"application/json"]],
                    }
                )
                error_response = {
                    "error": {
                        "detail": f"Access denied: user does not have access to this route. Access control policy: {webmethod.access_control}"
                    }
                }
                await send(
                    {
                        "type": "http.response.body",
                        "body": json.dumps(error_response).encode(),
                    }
                )
                return

        # Continue with the request
        return await self.app(scope, receive, send)


def _evaluate_policy(condition: str, user) -> bool:
    # if no user, assume auth is not enabled
    if not user:
        return True

    try:
        condition_obj = parse_condition(condition)

        if not (isinstance(condition_obj, UserWithValueInList) or isinstance(condition_obj, UserWithValueNotInList)):
            # Only support these two conditions
            return False

        # Create a dummy resource since we don't have one in this context
        class DummyResource:
            type = "api"
            identifier = "unknown"
            owner = user

        dummy_resource = DummyResource()

        return condition_obj.matches(dummy_resource, user)
    except (ValueError, AttributeError):
        return False
