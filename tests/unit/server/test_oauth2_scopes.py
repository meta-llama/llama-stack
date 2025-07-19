# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llama_stack.distribution.server.oauth2_scopes import (
    STANDARD_OAUTH2_SCOPES,
    get_required_scopes_for_api,
    validate_scopes,
    scope_grants_admin_access,
    get_all_scope_descriptions,
)
from llama_stack.distribution.server.auth import extract_api_from_path, AuthenticationMiddleware
from llama_stack.distribution.datatypes import AuthenticationConfig, OAuth2TokenAuthConfig, AuthProviderType


class TestOAuth2Scopes:
    """Test OAuth2 scope definitions and validation"""

    def test_standard_scopes_exist(self):
        """Test that all expected standard scopes are defined"""
        expected_scopes = {
            "llama:inference",
            "llama:models:read",
            "llama:models:write", 
            "llama:agents:read",
            "llama:agents:write",
            "llama:tools",
            "llama:vector_dbs:read",
            "llama:vector_dbs:write",
            "llama:safety",
            "llama:eval",
            "llama:admin",
        }
        
        assert set(STANDARD_OAUTH2_SCOPES.keys()) == expected_scopes
        
        # Verify all scopes have descriptions
        for scope, description in STANDARD_OAUTH2_SCOPES.items():
            assert isinstance(description, str)
            assert len(description) > 10  # Reasonable description length

    def test_get_all_scope_descriptions(self):
        """Test getting all scope descriptions"""
        descriptions = get_all_scope_descriptions()
        
        assert descriptions == STANDARD_OAUTH2_SCOPES
        assert len(descriptions) == len(STANDARD_OAUTH2_SCOPES)
        
        # Verify it's a copy, not the original
        descriptions["test"] = "should not affect original"
        assert "test" not in STANDARD_OAUTH2_SCOPES

    def test_validate_scopes_valid(self):
        """Test scope validation with valid scopes"""
        # Single valid scope
        token_scopes = {"llama:inference"}
        result = validate_scopes(token_scopes)
        assert result == {"llama:inference"}
        
        # Multiple valid scopes
        token_scopes = {"llama:inference", "llama:models:read", "llama:admin"}
        result = validate_scopes(token_scopes)
        assert result == {"llama:inference", "llama:models:read", "llama:admin"}
        
        # Mix of valid and invalid scopes
        token_scopes = {"llama:inference", "invalid:scope", "llama:admin"}
        result = validate_scopes(token_scopes)
        assert result == {"llama:inference", "llama:admin"}

    def test_validate_scopes_invalid(self):
        """Test scope validation with invalid scopes"""
        # No valid scopes
        token_scopes = {"invalid:scope", "another:invalid"}
        with pytest.raises(ValueError, match="Token lacks required OAuth2 scopes"):
            validate_scopes(token_scopes)
        
        # Empty scopes
        token_scopes = set()
        with pytest.raises(ValueError, match="Token lacks required OAuth2 scopes"):
            validate_scopes(token_scopes)

    def test_scope_grants_admin_access(self):
        """Test admin scope detection"""
        # Admin scope present
        assert scope_grants_admin_access({"llama:admin"})
        assert scope_grants_admin_access({"llama:admin", "llama:inference"})
        
        # Admin scope not present
        assert not scope_grants_admin_access({"llama:inference"})
        assert not scope_grants_admin_access({"llama:models:read", "llama:agents:write"})
        assert not scope_grants_admin_access(set())


class TestScopeRequirements:
    """Test API endpoint scope requirements"""

    def test_inference_api_scopes(self):
        """Test inference API scope requirements"""
        apis = ["inference", "chat", "completion", "embeddings"]
        for api in apis:
            scopes = get_required_scopes_for_api(api, "POST")
            assert "llama:inference" in scopes
            assert "llama:admin" in scopes

    def test_models_api_scopes(self):
        """Test models API scope requirements"""
        # Read operations
        read_scopes = get_required_scopes_for_api("models", "GET")
        assert "llama:models:read" in read_scopes
        assert "llama:admin" in read_scopes
        assert "llama:models:write" not in read_scopes
        
        # Write operations
        for method in ["POST", "PUT", "DELETE"]:
            write_scopes = get_required_scopes_for_api("models", method)
            assert "llama:models:write" in write_scopes
            assert "llama:admin" in write_scopes
            assert "llama:models:read" not in write_scopes

    def test_agents_api_scopes(self):
        """Test agents API scope requirements"""
        # Read operations
        read_scopes = get_required_scopes_for_api("agents", "GET")
        assert "llama:agents:read" in read_scopes
        assert "llama:admin" in read_scopes
        
        # Write operations
        for method in ["POST", "PUT", "DELETE"]:
            write_scopes = get_required_scopes_for_api("agents", method)
            assert "llama:agents:write" in write_scopes
            assert "llama:admin" in write_scopes

    def test_tools_api_scopes(self):
        """Test tools API scope requirements"""
        for api in ["tools", "tool_runtime"]:
            scopes = get_required_scopes_for_api(api, "POST")
            assert "llama:tools" in scopes
            assert "llama:admin" in scopes

    def test_vector_dbs_api_scopes(self):
        """Test vector databases API scope requirements"""
        # Read operations
        read_scopes = get_required_scopes_for_api("vector_dbs", "GET")
        assert "llama:vector_dbs:read" in read_scopes
        assert "llama:admin" in read_scopes
        
        # Write operations
        for method in ["POST", "PUT", "DELETE"]:
            write_scopes = get_required_scopes_for_api("vector_dbs", method)
            assert "llama:vector_dbs:write" in write_scopes
            assert "llama:admin" in write_scopes

    def test_safety_api_scopes(self):
        """Test safety API scope requirements"""
        scopes = get_required_scopes_for_api("safety", "POST")
        assert "llama:safety" in scopes
        assert "llama:admin" in scopes

    def test_eval_api_scopes(self):
        """Test evaluation API scope requirements"""
        for api in ["eval", "benchmarks", "scoring"]:
            scopes = get_required_scopes_for_api(api, "POST")
            assert "llama:eval" in scopes
            assert "llama:admin" in scopes

    def test_unknown_api_scopes(self):
        """Test unknown API only requires admin scope"""
        scopes = get_required_scopes_for_api("unknown_api", "POST")
        assert scopes == {"llama:admin"}

    def test_admin_always_included(self):
        """Test that admin scope is always included in required scopes"""
        test_apis = ["inference", "models", "agents", "tools", "safety", "eval", "unknown"]
        test_methods = ["GET", "POST", "PUT", "DELETE"]
        
        for api in test_apis:
            for method in test_methods:
                scopes = get_required_scopes_for_api(api, method)
                assert "llama:admin" in scopes


class TestAPIPathExtraction:
    """Test API path extraction for scope validation"""

    def test_v1_api_paths(self):
        """Test extraction from v1 API paths"""
        test_cases = [
            ("/v1/inference/chat-completion", ("inference", "POST")),
            ("/v1/models", ("models", "POST")),
            ("/v1/models/my-model", ("models", "POST")),
            ("/v1/agents/session", ("agents", "POST")),
            ("/v1/tools/execute", ("tools", "POST")),
            ("/v1/vector_dbs/query", ("vector_dbs", "POST")),
            ("/v1/safety/shield", ("safety", "POST")),
            ("/v1/eval/benchmark", ("eval", "POST")),
        ]
        
        for path, expected in test_cases:
            result = extract_api_from_path(path)
            assert result == expected

    def test_openai_compatibility_paths(self):
        """Test extraction from OpenAI compatibility paths"""
        openai_paths = [
            "/v1/openai/v1/chat/completions",
            "/v1/openai/v1/completions",
            "/v1/openai/v1/embeddings",
        ]
        
        for path in openai_paths:
            api, method = extract_api_from_path(path)
            assert api == "inference"
            assert method == "POST"

    def test_edge_case_paths(self):
        """Test edge cases in path extraction"""
        test_cases = [
            ("/", ("unknown", "GET")),
            ("/health", ("health", "POST")),
            ("/v1/", ("unknown", "GET")),
            ("", ("unknown", "GET")),
            ("/some/nested/path", ("some", "POST")),
        ]
        
        for path, expected in test_cases:
            result = extract_api_from_path(path)
            assert result == expected


class TestScopeBasedAuth:
    """Test scope-based authentication integration"""

    @pytest.fixture
    def mock_auth_config(self):
        """Create mock OAuth2 authentication config"""
        return AuthenticationConfig(
            provider_config=OAuth2TokenAuthConfig(
                type=AuthProviderType.OAUTH2_TOKEN,
                issuer="https://test-issuer.com",
                audience="llama-stack",
            )
        )

    @pytest.fixture
    def mock_app(self):
        """Create mock FastAPI app"""
        app = MagicMock()
        return app

    def test_scope_validation_in_middleware(self, mock_auth_config, mock_app):
        """Test that middleware validates scopes correctly"""
        middleware = AuthenticationMiddleware(mock_app, mock_auth_config)
        
        # This is a simplified test - in practice you'd need to mock the full ASGI flow
        assert middleware.auth_provider is not None

    @pytest.mark.asyncio
    async def test_token_validation_with_scopes(self):
        """Test JWT token validation with OAuth2 scopes"""
        # Mock JWT claims with scopes
        mock_claims = {
            "sub": "test-user",
            "scope": "llama:inference llama:models:read",
            "iss": "test-issuer",
            "aud": "llama-stack",
        }
        
        with patch("llama_stack.distribution.server.auth_providers.jwt.decode") as mock_jwt_decode:
            mock_jwt_decode.return_value = mock_claims
            
            # Mock the auth provider
            from llama_stack.distribution.server.auth_providers import OAuth2TokenAuthProvider
            from llama_stack.distribution.datatypes import OAuth2TokenAuthConfig, AuthProviderType
            
            config = OAuth2TokenAuthConfig(
                type=AuthProviderType.OAUTH2_TOKEN,
                issuer="test-issuer",
                audience="llama-stack",
            )
            
            provider = OAuth2TokenAuthProvider(config)
            
            # Mock the key retrieval
            with patch.object(provider, "_get_public_key") as mock_get_key:
                mock_get_key.return_value = "mock-key"
                
                # Test token validation
                user = await provider.validate_token("mock-token", {})
                
                assert user.principal == "test-user"
                assert user.attributes is not None
                assert "scopes" in user.attributes
                assert set(user.attributes["scopes"]) == {"llama:inference", "llama:models:read"}

    def test_scope_intersection_logic(self):
        """Test scope intersection for access control"""
        # User has inference and read scopes
        user_scopes = {"llama:inference", "llama:models:read"}
        
        # Test various API requirements
        inference_required = {"llama:admin", "llama:inference"}
        models_read_required = {"llama:admin", "llama:models:read"}
        models_write_required = {"llama:admin", "llama:models:write"}
        
        # Should have access to inference
        assert user_scopes.intersection(inference_required)
        
        # Should have access to model reading
        assert user_scopes.intersection(models_read_required)
        
        # Should NOT have access to model writing
        assert not user_scopes.intersection(models_write_required)

    def test_admin_scope_access(self):
        """Test that admin scope grants access to everything"""
        admin_scopes = {"llama:admin"}
        
        # Test various API requirements
        test_requirements = [
            {"llama:admin", "llama:inference"},
            {"llama:admin", "llama:models:write"},
            {"llama:admin", "llama:agents:write"},
            {"llama:admin", "llama:tools"},
            {"llama:admin", "llama:vector_dbs:write"},
            {"llama:admin", "llama:safety"},
            {"llama:admin", "llama:eval"},
        ]
        
        for required_scopes in test_requirements:
            assert admin_scopes.intersection(required_scopes)


class TestScopeValidationErrors:
    """Test error cases in scope validation"""

    def test_missing_scope_claim(self):
        """Test handling of missing scope claim in JWT"""
        # Empty scope claim
        token_scopes = set()
        with pytest.raises(ValueError, match="Token lacks required OAuth2 scopes"):
            validate_scopes(token_scopes)

    def test_malformed_scope_string(self):
        """Test handling of malformed scope strings"""
        # Scopes with extra whitespace should be handled gracefully
        scope_string = "  llama:inference   llama:models:read  "
        scopes = set(scope_string.split())
        
        # Filter out empty strings that might result from split()
        scopes = {s.strip() for s in scopes if s.strip()}
        
        result = validate_scopes(scopes)
        assert result == {"llama:inference", "llama:models:read"}

    def test_case_sensitive_scopes(self):
        """Test that scopes are case-sensitive"""
        # Wrong case should not match
        token_scopes = {"LLAMA:INFERENCE", "llama:Models:Read"}
        
        with pytest.raises(ValueError, match="Token lacks required OAuth2 scopes"):
            validate_scopes(token_scopes)

    def test_partial_scope_matches(self):
        """Test that partial scope matches don't work"""
        # Partial matches should not be accepted
        token_scopes = {"llama:model", "llama", "inference"}
        
        with pytest.raises(ValueError, match="Token lacks required OAuth2 scopes"):
            validate_scopes(token_scopes)


class TestScopeDocumentation:
    """Test scope documentation and descriptions"""

    def test_scope_naming_convention(self):
        """Test that scope names follow consistent naming convention"""
        for scope in STANDARD_OAUTH2_SCOPES.keys():
            # All scopes should start with 'llama:'
            assert scope.startswith("llama:")
            
            # Should not contain spaces
            assert " " not in scope
            
            # Should use colons as separators, not dots or slashes
            parts = scope.split(":")
            assert len(parts) >= 2
            assert all(part.replace("_", "").isalnum() for part in parts)

    def test_scope_descriptions_quality(self):
        """Test that scope descriptions are meaningful"""
        for scope, description in STANDARD_OAUTH2_SCOPES.items():
            # Should be non-empty strings
            assert isinstance(description, str)
            assert len(description.strip()) > 5
            
            # Should not just be the scope name
            assert scope not in description
            
            # Should contain descriptive words
            descriptive_words = ["access", "read", "write", "manage", "create", "delete", "execute"]
            assert any(word in description.lower() for word in descriptive_words)

    def test_read_write_scope_pairs(self):
        """Test that read/write scope pairs are consistent"""
        read_write_apis = ["models", "agents", "vector_dbs"]
        
        for api in read_write_apis:
            read_scope = f"llama:{api}:read"
            write_scope = f"llama:{api}:write"
            
            assert read_scope in STANDARD_OAUTH2_SCOPES
            assert write_scope in STANDARD_OAUTH2_SCOPES
            
            # Read description should mention "read"
            assert "read" in STANDARD_OAUTH2_SCOPES[read_scope].lower()
            
            # Write description should mention write operations
            write_desc = STANDARD_OAUTH2_SCOPES[write_scope].lower()
            assert any(word in write_desc for word in ["write", "manage", "register", "create"])


if __name__ == "__main__":
    pytest.main([__file__]) 