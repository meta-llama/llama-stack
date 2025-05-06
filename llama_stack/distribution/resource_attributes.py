# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.distribution.access_control import check_access
from llama_stack.distribution.datatypes import AccessAttributes, AccessAttributesRule, ResourceWithACL


def match_access_attributes_rule(
    rule: AccessAttributesRule, resource_type: str, resource_id: str, provider_id: str
) -> bool:
    if rule.resource_type and rule.resource_type.value != resource_type:
        return False
    if rule.resource_id and rule.resource_id != resource_id:
        return False
    if rule.provider_id and rule.provider_id != provider_id:
        return False
    return True


class ResourceAccessAttributes:
    def __init__(self, rules: list[AccessAttributesRule]) -> None:
        self.rules = rules
        self.access_check_enabled = False

    def enable_access_checks(self):
        self.access_check_enabled = True

    def get(self, resource_type: str, resource_id: str, provider_id: str) -> AccessAttributes | None:
        for rule in self.rules:
            if match_access_attributes_rule(rule, resource_type, resource_id, provider_id):
                return rule.attributes
        return None

    def apply(self, resource: ResourceWithACL, user_attributes: dict[str, list[str]] | None) -> bool:
        """Sets the resource access attributes based on the specified rules.

        Returns True if a matching rule was found for this resource.

        If access checks have been enable, also checks whether the user attributes allow the
        resource to be created.
        """

        resource_attributes = self.get(resource.type, resource.identifier, resource.provider_id)
        if resource_attributes:
            if self.access_check_enabled and not check_access(
                resource.identifier, resource_attributes, user_attributes
            ):
                raise ValueError(f"Access denied: {resource.type} '{resource.identifier}'")
            resource.access_attributes = resource_attributes
            return True
        return False
