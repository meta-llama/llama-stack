# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self


class AccessAttributes(BaseModel):
    """Structured representation of user attributes for access control.

    This model defines a structured approach to representing user attributes
    with common standard categories for access control.

    Standard attribute categories include:
    - roles: Role-based attributes (e.g., admin, data-scientist)
    - teams: Team-based attributes (e.g., ml-team, infra-team)
    - projects: Project access attributes (e.g., llama-3, customer-insights)
    - namespaces: Namespace-based access control for resource isolation
    """

    # Standard attribute categories - the minimal set we need now
    roles: list[str] | None = Field(
        default=None, description="Role-based attributes (e.g., 'admin', 'data-scientist', 'user')"
    )

    teams: list[str] | None = Field(default=None, description="Team-based attributes (e.g., 'ml-team', 'nlp-team')")

    projects: list[str] | None = Field(
        default=None, description="Project-based access attributes (e.g., 'llama-3', 'customer-insights')"
    )

    namespaces: list[str] | None = Field(
        default=None, description="Namespace-based access control for resource isolation"
    )


class Action(str, Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"


class Scope(BaseModel):
    principal: str | None = None
    actions: Action | list[Action]
    resource: str | None = None


def _mutually_exclusive(obj, a: str, b: str):
    if getattr(obj, a) and getattr(obj, b):
        raise ValueError(f"{a} and {b} are mutually exclusive")


def _require_one_of(obj, a: str, b: str):
    if not getattr(obj, a) and not getattr(obj, b):
        raise ValueError(f"on of {a} or {b} is required")


class AttributeReference(str, Enum):
    RESOURCE_ROLES = "resource.roles"
    RESOURCE_TEAMS = "resource.teams"
    RESOURCE_PROJECTS = "resource.projects"
    RESOURCE_NAMESPACES = "resource.namespaces"


class Condition(BaseModel):
    user_in: AttributeReference | list[AttributeReference] | str | None = None
    user_not_in: AttributeReference | list[AttributeReference] | str | None = None


class AccessRule(BaseModel):
    """Access rule based loosely on cedar policy language

    A rule defines a list of action either to permit or to forbid. It may specify a
    principal or a resource that must match for the rule to take effect. The resource
    to match should be specified in the form of a type qualified identifier, e.g.
    model::my-model or vector_db::some-db, or a wildcard for all resources of a type,
    e.g. model::*. If the principal or resource are not specified, they will match all
    requests.

    A rule may also specify a condition, either a 'when' or an 'unless', with additional
    constraints as to where the rule applies. The constraints at present are whether the
    user requesting access is in or not in some set. This set can either be a particular
    set of attributes on the resource e.g. resource.roles or a literal value of some
    notion of group, e.g. role::admin or namespace::foo.

    Rules are tested in order to find a match. If a match is found, the request is
    permitted or forbidden depending on the type of rule. If no match is found, the
    request is denied. If no rules are specified, a rule that allows any action as
    long as the resource attributes match the user attributes is added
    (i.e. the previous behaviour is the default).

    Some examples in yaml:

    - permit:
      principal: user-1
      actions: [create, read, delete]
      resource: model::*
      description: user-1 has full access to all models
    - permit:
      principal: user-2
      actions: [read]
      resource: model::model-1
      description: user-2 has read access to model-1 only
    - permit:
      actions: [read]
      when:
        user_in: resource.namespaces
      description: any user has read access to any resource with matching attributes
    - forbid:
      actions: [create, read, delete]
      resource: vector_db::*
      unless:
        user_in: role::admin
      description: only user with admin role can use vector_db resources

    """

    permit: Scope | None = None
    forbid: Scope | None = None
    when: Condition | list[Condition] | None = None
    unless: Condition | list[Condition] | None = None
    description: str | None = None

    @model_validator(mode="after")
    def validate_rule_format(self) -> Self:
        _require_one_of(self, "permit", "forbid")
        _mutually_exclusive(self, "permit", "forbid")
        _mutually_exclusive(self, "when", "unless")
        return self
