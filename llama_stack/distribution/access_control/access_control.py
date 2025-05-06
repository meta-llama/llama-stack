# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol

from llama_stack.distribution.request_headers import User

from .datatypes import (
    AccessAttributes,
    AccessRule,
    Action,
    AttributeReference,
    Condition,
    Scope,
)


def matches_resource(resource_scope: str, actual_resource: str) -> bool:
    if resource_scope == actual_resource:
        return True
    return resource_scope.endswith("::*") and actual_resource.startswith(resource_scope[:-1])


def matches_scope(
    scope: Scope,
    action: Action,
    resource: str,
    user: str | None,
) -> bool:
    if scope.resource and not matches_resource(scope.resource, resource):
        return False
    if scope.principal and scope.principal != user:
        return False
    return action in scope.actions


def user_in_literal(
    literal: str,
    user_attributes: dict[str, list[str]] | None,
) -> bool:
    for qualifier in ["role::", "team::", "project::", "namespace::"]:
        if literal.startswith(qualifier):
            if not user_attributes:
                return False
            ref = qualifier.replace("::", "s")
            if ref in user_attributes:
                value = literal.removeprefix(qualifier)
                return value in user_attributes[ref]
            else:
                return False
    return False


def user_in(
    ref: AttributeReference | str,
    resource_attributes: AccessAttributes | None,
    user_attributes: dict[str, list[str]] | None,
) -> bool:
    if not ref.startswith("resource."):
        return user_in_literal(ref, user_attributes)
    name = ref.removeprefix("resource.")
    required = resource_attributes and getattr(resource_attributes, name)
    if not required:
        return True
    if not user_attributes or name not in user_attributes:
        return False
    actual = user_attributes[name]
    for value in required:
        if value in actual:
            return True
    return False


def as_list(obj: Any) -> list[Any]:
    if isinstance(obj, list):
        return obj
    return [obj]


def matches_conditions(
    conditions: list[Condition],
    resource_attributes: AccessAttributes | None,
    user_attributes: dict[str, list[str]] | None,
) -> bool:
    for condition in conditions:
        # must match all conditions
        if not matches_condition(condition, resource_attributes, user_attributes):
            return False
    return True


def matches_condition(
    condition: Condition | list[Condition],
    resource_attributes: AccessAttributes | None,
    user_attributes: dict[str, list[str]] | None,
) -> bool:
    if isinstance(condition, list):
        return matches_conditions(as_list(condition), resource_attributes, user_attributes)
    if condition.user_in:
        for ref in as_list(condition.user_in):
            # if multiple references are specified, all must match
            if not user_in(ref, resource_attributes, user_attributes):
                return False
        return True
    if condition.user_not_in:
        for ref in as_list(condition.user_not_in):
            # if multiple references are specified, none must match
            if user_in(ref, resource_attributes, user_attributes):
                return False
        return True
    return True


def default_policy() -> list[AccessRule]:
    # for backwards compatibility, if no rules are provided , assume
    # full access to all subject to attribute matching rules
    return [
        AccessRule(
            permit=Scope(actions=list(Action)),
            when=Condition(user_in=list(AttributeReference)),
        )
    ]


class ProtectedResource(Protocol):
    type: str
    identifier: str
    access_attributes: AccessAttributes


def is_action_allowed(
    policy: list[AccessRule],
    action: Action,
    resource: ProtectedResource,
    user: User | None,
) -> bool:
    # If user is not set, assume authentication is not enabled
    if not user:
        return True

    if not len(policy):
        policy = default_policy()

    resource_attributes = AccessAttributes()
    if hasattr(resource, "access_attributes"):
        resource_attributes = resource.access_attributes
    qualified_resource_id = resource.type + "::" + resource.identifier
    for rule in policy:
        if rule.forbid and matches_scope(rule.forbid, action, qualified_resource_id, user.principal):
            if rule.when:
                if matches_condition(rule.when, resource_attributes, user.attributes):
                    return False
            elif rule.unless:
                if not matches_condition(rule.unless, resource_attributes, user.attributes):
                    return False
            else:
                return False
        elif rule.permit and matches_scope(rule.permit, action, qualified_resource_id, user.principal):
            if rule.when:
                if matches_condition(rule.when, resource_attributes, user.attributes):
                    return True
            elif rule.unless:
                if not matches_condition(rule.unless, resource_attributes, user.attributes):
                    return True
            else:
                return True
    # assume access is denied unless we find a rule that permits access
    return False


class AccessDeniedError(RuntimeError):
    pass
