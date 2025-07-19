# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
OAuth2 scope definitions and validation for Llama Stack APIs.

This module defines the standard OAuth2 scopes that are built-in to Llama Stack
and provides utilities for scope validation. These scopes are not configurable
and provide a standardized way to control access to different API endpoints.
"""

from typing import Set

# Standard OAuth2 scopes for Llama Stack APIs (built-in, not configurable)
STANDARD_OAUTH2_SCOPES = {
    # Inference API
    "llama:inference": "Access to inference APIs (chat completion, embeddings)",
    
    # Models API  
    "llama:models:read": "Read access to models (list, get model details)",
    "llama:models:write": "Write access to models (register, unregister)",
    
    # Agents API
    "llama:agents:read": "Read access to agents (list sessions, get agent details)", 
    "llama:agents:write": "Write access to agents (create sessions, send messages)",
    
    # Tools API
    "llama:tools": "Access to tool runtime and execution",
    
    # Vector DB API
    "llama:vector_dbs:read": "Read access to vector databases",
    "llama:vector_dbs:write": "Write access to vector databases",
    
    # Safety API
    "llama:safety": "Access to safety shields and content filtering",
    
    # Eval API  
    "llama:eval": "Access to evaluation and benchmarking",
    
    # Administrative access
    "llama:admin": "Full administrative access to all APIs",
}


def get_required_scopes_for_api(api_name: str, method: str = "GET") -> Set[str]:
    """Get required OAuth2 scopes for accessing a specific API endpoint.
    
    Args:
        api_name: The name of the API (e.g., 'models', 'inference', 'agents')
        method: The HTTP method (GET, POST, PUT, DELETE)
        
    Returns:
        Set of scope strings that would grant access to this endpoint.
        Always includes 'llama:admin' as it grants access to everything.
    """
    # Admin scope grants access to everything
    required_scopes = {"llama:admin"}
    
    # Map API names to required scopes
    if api_name in ["inference", "chat", "completion", "embeddings"]:
        required_scopes.add("llama:inference")
    elif api_name == "models":
        if method in ["POST", "PUT", "DELETE"]:
            required_scopes.add("llama:models:write") 
        else:
            required_scopes.add("llama:models:read")
    elif api_name == "agents":
        if method in ["POST", "PUT", "DELETE"]:
            required_scopes.add("llama:agents:write")
        else:
            required_scopes.add("llama:agents:read") 
    elif api_name in ["tools", "tool_runtime"]:
        required_scopes.add("llama:tools")
    elif api_name == "vector_dbs":
        if method in ["POST", "PUT", "DELETE"]:
            required_scopes.add("llama:vector_dbs:write")
        else:
            required_scopes.add("llama:vector_dbs:read")
    elif api_name == "safety":
        required_scopes.add("llama:safety")
    elif api_name in ["eval", "benchmarks", "scoring"]:
        required_scopes.add("llama:eval")
    
    return required_scopes


def validate_scopes(token_scopes: Set[str]) -> Set[str]:
    """Validate OAuth2 scopes against standard Llama Stack scopes.
    
    Args:
        token_scopes: Set of scopes from the OAuth2 token
        
    Returns:
        Set of valid scopes that are recognized by Llama Stack
        
    Raises:
        ValueError: If no valid scopes are found
    """
    valid_scopes = token_scopes.intersection(STANDARD_OAUTH2_SCOPES.keys())
    if not valid_scopes:
        raise ValueError("Token lacks required OAuth2 scopes for Llama Stack access")
    return valid_scopes


def scope_grants_admin_access(scopes: Set[str]) -> bool:
    """Check if the provided scopes include administrative access.
    
    Args:
        scopes: Set of OAuth2 scopes
        
    Returns:
        True if scopes include administrative access, False otherwise
    """
    return "llama:admin" in scopes


def get_all_scope_descriptions() -> dict[str, str]:
    """Get all standard OAuth2 scopes with their descriptions.
    
    Returns:
        Dictionary mapping scope names to their descriptions
    """
    return STANDARD_OAUTH2_SCOPES.copy() 