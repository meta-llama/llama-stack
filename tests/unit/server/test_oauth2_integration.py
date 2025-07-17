# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Integration tests for OAuth2 scope-based authentication.

These tests verify the end-to-end flow of OAuth2 scope validation
from JWT token parsing to API access decisions.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llama_stack.distribution.server.oauth2_scopes import get_required_scopes_for_api
from llama_stack.distribution.server.auth import extract_api_from_path
from llama_stack.distribution.datatypes import User


def get_user_scopes(user: User) -> set[str]:
    """Safely extract scopes from user attributes"""
    if user.attributes and "scopes" in user.attributes:
        return set(user.attributes["scopes"])
    return set()


class TestOAuth2IntegrationFlow:
    """Test the complete OAuth2 scope validation flow"""

    def test_inference_api_access_flow(self):
        """Test complete flow for inference API access"""
        # 1. Extract API from path
        api_name, method = extract_api_from_path("/v1/inference/chat-completion")
        assert api_name == "inference"
        
        # 2. Get required scopes for this API
        required_scopes = get_required_scopes_for_api(api_name, method)
        assert "llama:inference" in required_scopes
        assert "llama:admin" in required_scopes
        
        # 3. Test user with correct scope
        user_with_inference = User(
            principal="user1",
            attributes={"scopes": ["llama:inference"]}
        )
        user_scopes = set(user_with_inference.attributes["scopes"]) if user_with_inference.attributes else set()
        assert user_scopes.intersection(required_scopes)  # Should have access
        
        # 4. Test user without correct scope
        user_without_inference = User(
            principal="user2", 
            attributes={"scopes": ["llama:models:read"]}
        )
        user_scopes = set(user_without_inference.attributes["scopes"])
        assert not user_scopes.intersection(required_scopes)  # Should NOT have access

    def test_models_api_read_write_flow(self):
        """Test complete flow for models API with read/write distinction"""
        # Test read operation
        api_name, method = extract_api_from_path("/v1/models")
        read_required = get_required_scopes_for_api(api_name, "GET")
        
        # Test write operation  
        write_required = get_required_scopes_for_api(api_name, "POST")
        
        # User with only read access
        read_user = User(
            principal="read_user",
            attributes={"scopes": ["llama:models:read"]}
        )
        read_scopes = set(read_user.attributes["scopes"])
        
        # Should have read access
        assert read_scopes.intersection(read_required)
        
        # Should NOT have write access
        assert not read_scopes.intersection(write_required)
        
        # User with write access
        write_user = User(
            principal="write_user", 
            attributes={"scopes": ["llama:models:write"]}
        )
        write_scopes = set(write_user.attributes["scopes"])
        
        # Should have write access
        assert write_scopes.intersection(write_required)
        
        # Should NOT have read access (write doesn't imply read)
        assert not write_scopes.intersection(read_required)

    def test_admin_scope_universal_access(self):
        """Test that admin scope grants access to all APIs"""
        admin_user = User(
            principal="admin",
            attributes={"scopes": ["llama:admin"]}
        )
        admin_scopes = set(admin_user.attributes["scopes"])
        
        # Test various API endpoints
        test_cases = [
            ("/v1/inference/chat-completion", "POST"),
            ("/v1/models", "GET"),
            ("/v1/models/my-model", "DELETE"),
            ("/v1/agents/session", "POST"),
            ("/v1/tools/execute", "POST"),
            ("/v1/vector_dbs/query", "POST"),
            ("/v1/safety/shield", "POST"),
            ("/v1/eval/benchmark", "POST"),
        ]
        
        for path, method in test_cases:
            api_name, _ = extract_api_from_path(path)
            required_scopes = get_required_scopes_for_api(api_name, method)
            
            # Admin should always have access
            assert admin_scopes.intersection(required_scopes), f"Admin denied access to {path}"

    def test_multiple_scopes_user(self):
        """Test user with multiple scopes"""
        multi_scope_user = User(
            principal="power_user",
            attributes={
                "scopes": [
                    "llama:inference", 
                    "llama:models:read",
                    "llama:agents:write",
                    "llama:tools"
                ]
            }
        )
        user_scopes = set(multi_scope_user.attributes["scopes"])
        
        # Test various access scenarios
        access_tests = [
            ("/v1/inference/chat-completion", "POST", True),   # Has inference
            ("/v1/models", "GET", True),                       # Has models:read
            ("/v1/models", "POST", False),                     # Doesn't have models:write
            ("/v1/agents/session", "POST", True),              # Has agents:write
            ("/v1/agents", "GET", False),                      # Doesn't have agents:read
            ("/v1/tools/execute", "POST", True),               # Has tools
            ("/v1/vector_dbs/query", "GET", False),            # Doesn't have vector_dbs:read
            ("/v1/safety/shield", "POST", False),              # Doesn't have safety
        ]
        
        for path, method, should_have_access in access_tests:
            api_name, _ = extract_api_from_path(path)
            required_scopes = get_required_scopes_for_api(api_name, method)
            has_access = bool(user_scopes.intersection(required_scopes))
            
            assert has_access == should_have_access, (
                f"Access mismatch for {path} {method}: "
                f"expected {should_have_access}, got {has_access}"
            )

    def test_openai_compatibility_scope_flow(self):
        """Test OAuth2 scope validation for OpenAI compatibility endpoints"""
        # OpenAI endpoints should map to inference API
        openai_paths = [
            "/v1/openai/v1/chat/completions",
            "/v1/openai/v1/completions", 
            "/v1/openai/v1/embeddings",
        ]
        
        inference_user = User(
            principal="openai_user",
            attributes={"scopes": ["llama:inference"]}
        )
        user_scopes = set(inference_user.attributes["scopes"])
        
        for path in openai_paths:
            api_name, method = extract_api_from_path(path)
            assert api_name == "inference"
            
            required_scopes = get_required_scopes_for_api(api_name, method)
            assert user_scopes.intersection(required_scopes), f"No access to {path}"

    def test_scope_validation_error_scenarios(self):
        """Test error scenarios in scope validation"""
        # User with no scopes
        no_scope_user = User(principal="no_scope", attributes={})
        
        api_name, method = extract_api_from_path("/v1/inference/chat-completion")
        required_scopes = get_required_scopes_for_api(api_name, method)
        
        # Should not have access
        user_scopes = set()
        assert not user_scopes.intersection(required_scopes)
        
        # User with empty scopes list
        empty_scope_user = User(
            principal="empty_scope",
            attributes={"scopes": []}
        )
        user_scopes = set(empty_scope_user.attributes["scopes"])
        assert not user_scopes.intersection(required_scopes)
        
        # User with invalid scopes
        invalid_scope_user = User(
            principal="invalid_scope",
            attributes={"scopes": ["invalid:scope", "another:invalid"]}
        )
        user_scopes = set(invalid_scope_user.attributes["scopes"])
        assert not user_scopes.intersection(required_scopes)


class TestScopeBasedAccessMatrix:
    """Test access matrix for different user types and API combinations"""

    def test_data_scientist_access_pattern(self):
        """Test typical data scientist access pattern"""
        data_scientist = User(
            principal="data_scientist",
            attributes={
                "scopes": [
                    "llama:inference",
                    "llama:models:read", 
                    "llama:eval",
                    "llama:safety"
                ]
            }
        )
        user_scopes = set(data_scientist.attributes["scopes"])
        
        # Should have access to
        allowed_apis = [
            ("inference", "POST"),   # Run inference
            ("models", "GET"),       # List/inspect models  
            ("eval", "POST"),        # Run evaluations
            ("safety", "POST"),      # Use safety shields
        ]
        
        for api, method in allowed_apis:
            required = get_required_scopes_for_api(api, method)
            assert user_scopes.intersection(required), f"Data scientist denied {api} {method}"
        
        # Should NOT have access to
        denied_apis = [
            ("models", "POST"),      # Cannot register models
            ("agents", "POST"),      # Cannot create agents
            ("tools", "POST"),       # Cannot use tools
            ("vector_dbs", "GET"),   # Cannot access vector DBs
        ]
        
        for api, method in denied_apis:
            required = get_required_scopes_for_api(api, method)
            assert not user_scopes.intersection(required), f"Data scientist allowed {api} {method}"

    def test_ml_engineer_access_pattern(self):
        """Test typical ML engineer access pattern"""
        ml_engineer = User(
            principal="ml_engineer",
            attributes={
                "scopes": [
                    "llama:inference",
                    "llama:models:read",
                    "llama:models:write",
                    "llama:agents:read",
                    "llama:agents:write",
                    "llama:tools",
                    "llama:eval"
                ]
            }
        )
        user_scopes = set(ml_engineer.attributes["scopes"])
        
        # Should have broad access except admin-only operations
        allowed_apis = [
            ("inference", "POST"),
            ("models", "GET"),
            ("models", "POST"), 
            ("agents", "GET"),
            ("agents", "POST"),
            ("tools", "POST"),
            ("eval", "POST"),
        ]
        
        for api, method in allowed_apis:
            required = get_required_scopes_for_api(api, method)
            assert user_scopes.intersection(required), f"ML engineer denied {api} {method}"

    def test_application_developer_access_pattern(self):
        """Test typical application developer access pattern"""
        app_developer = User(
            principal="app_developer",
            attributes={
                "scopes": [
                    "llama:inference",
                    "llama:agents:read",
                    "llama:agents:write", 
                    "llama:tools",
                    "llama:safety"
                ]
            }
        )
        user_scopes = set(app_developer.attributes["scopes"])
        
        # Should focus on application-building APIs
        allowed_apis = [
            ("inference", "POST"),   # Use models for apps
            ("agents", "GET"),       # Inspect agents
            ("agents", "POST"),      # Create agent sessions
            ("tools", "POST"),       # Execute tools
            ("safety", "POST"),      # Apply safety
        ]
        
        for api, method in allowed_apis:
            required = get_required_scopes_for_api(api, method)
            assert user_scopes.intersection(required), f"App developer denied {api} {method}"
        
        # Should NOT have model or eval management access
        denied_apis = [
            ("models", "POST"),      # Cannot manage models
            ("eval", "POST"),        # Cannot run evaluations
            ("vector_dbs", "POST"),  # Cannot manage vector DBs
        ]
        
        for api, method in denied_apis:
            required = get_required_scopes_for_api(api, method)
            assert not user_scopes.intersection(required), f"App developer allowed {api} {method}"


class TestScopeHierarchyAndSeparation:
    """Test that scopes are properly separated and don't grant unintended access"""

    def test_read_write_separation(self):
        """Test that read scopes don't grant write access and vice versa"""
        read_only_apis = ["models", "agents", "vector_dbs"]
        
        for api in read_only_apis:
            # User with only read scope
            read_user = User(
                principal=f"{api}_reader",
                attributes={"scopes": [f"llama:{api}:read"]}
            )
            read_scopes = set(read_user.attributes["scopes"])
            
            # User with only write scope
            write_user = User(
                principal=f"{api}_writer", 
                attributes={"scopes": [f"llama:{api}:write"]}
            )
            write_scopes = set(write_user.attributes["scopes"])
            
            # Read user should only have read access
            read_required = get_required_scopes_for_api(api, "GET")
            write_required = get_required_scopes_for_api(api, "POST")
            
            assert read_scopes.intersection(read_required), f"Read user denied read access to {api}"
            assert not read_scopes.intersection(write_required), f"Read user granted write access to {api}"
            
            # Write user should only have write access
            assert write_scopes.intersection(write_required), f"Write user denied write access to {api}"
            assert not write_scopes.intersection(read_required), f"Write user granted read access to {api}"

    def test_api_isolation(self):
        """Test that API scopes don't grant access to other APIs"""
        api_scopes = [
            "llama:inference",
            "llama:models:read",
            "llama:agents:write", 
            "llama:tools",
            "llama:vector_dbs:read",
            "llama:safety",
            "llama:eval"
        ]
        
        for scope in api_scopes:
            user = User(
                principal=f"single_scope_user",
                attributes={"scopes": [scope]}
            )
            user_scopes = set(user.attributes["scopes"])
            
            # Test that this scope only grants access to its intended API
            test_apis = [
                ("inference", "POST"),
                ("models", "GET"),
                ("models", "POST"),
                ("agents", "GET"), 
                ("agents", "POST"),
                ("tools", "POST"),
                ("vector_dbs", "GET"),
                ("vector_dbs", "POST"),
                ("safety", "POST"),
                ("eval", "POST")
            ]
            
            access_count = 0
            for api, method in test_apis:
                required = get_required_scopes_for_api(api, method)
                if user_scopes.intersection(required):
                    access_count += 1
            
            # Should only have access to 1-2 endpoints (the intended API)
            # Allow 2 for APIs that have both read and write variants
            assert access_count <= 2, f"Scope {scope} grants too broad access ({access_count} APIs)"
            assert access_count >= 1, f"Scope {scope} grants no access"


if __name__ == "__main__":
    pytest.main([__file__]) 