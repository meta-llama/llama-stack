# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from llama_stack.apis.datasets import URIDataSource
from llama_stack.distribution.datatypes import (
    AuthenticationConfig,
    DatasetWithACL,
    ModelWithACL,
    ShieldWithACL,
)
from llama_stack.distribution.resource_attributes import ResourceAccessAttributes, match_access_attributes_rule


@pytest.fixture
def rules():
    config = """{
    "provider_type": "custom",
    "config": {},
    "resource_attribute_rules": [
        {
            "resource_type": "model",
            "resource_id": "my-model",
            "provider_id": "my-provider",
            "attributes": {
                "roles": ["role1"],
                "teams": ["team1"],
                "projects": ["project1"],
                "namespaces": ["namespace1"]
            }
        },
        {
            "resource_type": "dataset",
            "provider_id": "my-provider",
            "attributes": {
                "roles": ["role2"],
                "teams": ["team2"],
                "projects": ["project2"],
                "namespaces": ["namespace2"]
            }
        },
        {
            "provider_id": "my-provider",
            "attributes": {
                "roles": ["role3"],
                "teams": ["team3"],
                "projects": ["project3"],
                "namespaces": ["namespace3"]
            }
        },
        {
            "resource_type": "model",
            "attributes": {
                "roles": ["role4"],
                "teams": ["team4"],
                "projects": ["project4"],
                "namespaces": ["namespace4"]
            }
        },
        {
            "attributes": {
                "roles": ["role5"],
                "teams": ["team5"],
                "projects": ["project5"],
                "namespaces": ["namespace5"]
            }
        }
    ]
}"""
    return AuthenticationConfig.model_validate_json(config).resource_attribute_rules


def test_match_access_attributes_rule(rules):
    assert match_access_attributes_rule(rules[0], "model", "my-model", "my-provider")
    assert not match_access_attributes_rule(rules[0], "model", "another-model", "my-provider")
    assert not match_access_attributes_rule(rules[0], "model", "my-model", "another-provider")
    assert not match_access_attributes_rule(rules[0], "dataset", "my-model", "my-provider")

    assert match_access_attributes_rule(rules[1], "dataset", "my-data", "my-provider")
    assert match_access_attributes_rule(rules[1], "dataset", "different-data", "my-provider")
    assert match_access_attributes_rule(rules[1], "dataset", "any-data", "my-provider")
    assert not match_access_attributes_rule(rules[1], "model", "a-model", "my-provider")
    assert not match_access_attributes_rule(rules[1], "dataset", "any-data", "another-provider")
    assert not match_access_attributes_rule(rules[1], "model", "my-model", "my-provider")

    assert match_access_attributes_rule(rules[2], "dataset", "foo", "my-provider")
    assert match_access_attributes_rule(rules[2], "model", "foo", "my-provider")
    assert match_access_attributes_rule(rules[2], "vector_db", "bar", "my-provider")
    assert not match_access_attributes_rule(rules[2], "dataset", "foo", "another-provider")

    assert match_access_attributes_rule(rules[3], "model", "foo", "my-provider")
    assert match_access_attributes_rule(rules[3], "model", "bar", "my-provider")
    assert match_access_attributes_rule(rules[3], "model", "foo", "another-provider")
    assert not match_access_attributes_rule(rules[3], "dataset", "bar", "my-provider")

    assert match_access_attributes_rule(rules[4], "model", "foo", "my-provider")
    assert match_access_attributes_rule(rules[4], "model", "bar", "my-provider")
    assert match_access_attributes_rule(rules[4], "model", "foo", "another-provider")
    assert match_access_attributes_rule(rules[4], "dataset", "bar", "my-provider")
    assert match_access_attributes_rule(rules[4], "vector_db", "baz", "any-provider")


@pytest.fixture
def resource_access_attributes(rules):
    return ResourceAccessAttributes(rules)


def dataset(identifier: str, provider_id: str) -> DatasetWithACL:
    return DatasetWithACL(
        identifier=identifier,
        provider_id=provider_id,
        purpose="eval/question-answer",
        source=URIDataSource(uri="https://a.com/a.jsonl"),
    )


@pytest.mark.parametrize(
    "resource,expected_roles",
    [
        (ModelWithACL(identifier="my-model", provider_id="my-provider"), ["role1"]),
        (ModelWithACL(identifier="another-model", provider_id="my-provider"), ["role3"]),
        (ModelWithACL(identifier="third-model", provider_id="another-provider"), ["role4"]),
        (dataset(identifier="my-dataset", provider_id="my-provider"), ["role2"]),
        (dataset(identifier="another-dataset", provider_id="another-provider"), ["role5"]),
        (ShieldWithACL(identifier="my-shield", provider_id="my-provider"), ["role3"]),
        (ShieldWithACL(identifier="another-shield", provider_id="another-provider"), ["role5"]),
    ],
)
def test_apply(resource_access_attributes, resource, expected_roles):
    assert resource_access_attributes.apply(resource, None)
    assert resource.access_attributes.roles == expected_roles


@pytest.fixture
def alternate_access_attributes_rules():
    config = """{
    "provider_type": "custom",
    "config": {},
    "resource_attribute_rules": [
        {
            "resource_type": "model",
            "resource_id": "my-model",
            "provider_id": "my-provider",
            "attributes": {
                "roles": ["roleA"]
            }
        },
        {
            "resource_type": "model",
            "attributes": {
                "roles": ["roleB"]
            }
        }
    ]
}"""
    return AuthenticationConfig.model_validate_json(config).resource_attribute_rules


@pytest.fixture
def alternate_access_attributes(alternate_access_attributes_rules):
    return ResourceAccessAttributes(alternate_access_attributes_rules)


@pytest.mark.parametrize(
    "resource,expected_roles",
    [
        (ModelWithACL(identifier="my-model", provider_id="my-provider"), ["roleA"]),
        (ModelWithACL(identifier="another-model", provider_id="another-provider"), ["roleB"]),
        (dataset(identifier="my-dataset", provider_id="my-provider"), None),
        (dataset(identifier="another-dataset", provider_id="another-provider"), None),
    ],
)
def test_apply_alternate(alternate_access_attributes, resource, expected_roles):
    if expected_roles:
        assert alternate_access_attributes.apply(resource, None)
        assert resource.access_attributes.roles == expected_roles
    else:
        assert not alternate_access_attributes.apply(resource, None)
        assert not resource.access_attributes


@pytest.fixture
def checked_attributes(alternate_access_attributes_rules):
    attributes = ResourceAccessAttributes(alternate_access_attributes_rules)
    attributes.enable_access_checks()
    return attributes


@pytest.mark.parametrize(
    "resource,user_attributes,fails,result",
    [
        (ModelWithACL(identifier="my-model", provider_id="my-provider"), {"roles": ["roleA"]}, False, True),
        (ModelWithACL(identifier="my-model", provider_id="my-provider"), {"roles": ["somethingelse"]}, True, False),
        (dataset(identifier="my-dataset", provider_id="my-provider"), {"roles": ["somethingelse"]}, False, False),
        (dataset(identifier="my-dataset", provider_id="my-provider"), None, False, False),
    ],
)
def test_access_check_on_apply(checked_attributes, resource, user_attributes, fails, result):
    if fails:
        with pytest.raises(ValueError) as e:
            checked_attributes.apply(resource, user_attributes)
        assert "Access denied" in str(e.value)
        assert not resource.access_attributes
    else:
        assert checked_attributes.apply(resource, user_attributes) == result
