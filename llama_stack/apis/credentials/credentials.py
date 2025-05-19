# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from llama_stack.schema_utils import json_schema_type, webmethod


class CredentialTokenType(str, Enum):
    """The type of credential token.

    :cvar oauth2: OAuth2 token
    :cvar api_key: API key

    """

    oauth2 = "oauth2"
    api_key = "api_key"


@json_schema_type
class ProviderCredential(BaseModel):
    credential_id: str
    provider_id: str
    token_type: CredentialTokenType
    token: str
    expires_at: datetime


@runtime_checkable
class Credentials(Protocol):
    """
    Create, update and delete ephemeral provider-specific credentials.

    Each provider may need optional authentication. This might be a persistent API key, or
    a short-lived OAuth2 token. There is a single credential for each provider instance.

    Credentials are ephemeral -- they may be purged after the specified TTL.

    Credentials are associated with the same ABAC access attributes and permissions as other
    resources in the system.

    It is recommended to store these credentials using Envelope Encryption. The storage could
    be a regular KVStore, but you should use a secure Key Management Service for encrypting
    and decrypting.
    """

    @webmethod(route="/credentials", method="POST")
    async def create_credential(
        self, provider_id: str, token_type: CredentialTokenType, token: str, ttl_seconds: int = 3600
    ) -> ProviderCredential:
        """Create a new set of credentials for a given provider.

        :param provider_id: The ID of the provider to create credentials for.
        :param token_type: The type of token to create. This is provided in the API to serve as lightweight
                           documentation / metadata for the token.
        :param token: The token itself.
        :param ttl_seconds: The time to live for the credential in seconds. Defaults to 3600 seconds.
        :returns: created ProviderCredential object
        """
        ...

    @webmethod(route="/credentials/{credential_id}", method="PUT")
    async def update_credential(self, credential_id: str, token: str) -> ProviderCredential:
        """Update an existing set of credentials for a given provider.

        :param credential_id: The ID of the credential to update.
        :param token: The new token to set for the credential.
        :returns: updated ProviderCredential object
        """
        ...

    @webmethod(route="/credentials/{credential_id}", method="DELETE")
    async def delete_credential(self, credential_id: str) -> None:
        """Delete a credential by its ID.

        :param credential_id: The ID of the credential to delete.
        """
        ...
