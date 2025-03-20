# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, Optional

from llama_stack.distribution.datatypes import RoutableObjectWithProvider
from llama_stack.log import get_logger

logger = get_logger(__name__, category="core")


def check_access(obj: RoutableObjectWithProvider, user_attributes: Optional[Dict[str, Any]] = None) -> bool:
    """Check if the current user has access to the given object, based on access attributes.

    Access control algorithm:
    1. If the resource has no access_attributes, access is GRANTED to all authenticated users
    2. If the user has no attributes, access is DENIED to any object with access_attributes defined
    3. For each attribute category in the resource's access_attributes:
       a. If the user lacks that category, access is DENIED
       b. If the user has the category but none of the required values, access is DENIED
       c. If the user has at least one matching value in each required category, access is GRANTED

    Example:
        # Resource requires:
        access_attributes = AccessAttributes(
            roles=["admin", "data-scientist"],
            teams=["ml-team"]
        )

        # User has:
        user_attributes = {
            "roles": ["data-scientist", "engineer"],
            "teams": ["ml-team", "infra-team"],
            "projects": ["llama-3"]
        }

        # Result: Access GRANTED
        # - User has the "data-scientist" role (matches one of the required roles)
        # - AND user is part of the "ml-team" (matches the required team)
        # - The extra "projects" attribute is ignored

    Args:
        obj: The resource object to check access for

    Returns:
        bool: True if access is granted, False if denied
    """
    # If object has no access attributes, allow access by default
    if not hasattr(obj, "access_attributes") or not obj.access_attributes:
        return True

    # If no user attributes, deny access to objects with access control
    if not user_attributes:
        return False

    obj_attributes = obj.access_attributes.model_dump(exclude_none=True)
    if not obj_attributes:
        return True

    # Check each attribute category (requires ALL categories to match)
    for attr_key, required_values in obj_attributes.items():
        user_values = user_attributes.get(attr_key, [])

        if not user_values:
            logger.debug(
                f"Access denied to {obj.type} '{obj.identifier}': missing required attribute category '{attr_key}'"
            )
            return False

        if not any(val in user_values for val in required_values):
            logger.debug(
                f"Access denied to {obj.type} '{obj.identifier}': "
                f"no match for attribute '{attr_key}', required one of {required_values}"
            )
            return False

    logger.debug(f"Access granted to {obj.type} '{obj.identifier}'")
    return True
