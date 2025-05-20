# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llama_stack.schema_utils import webmethod


class CredentialTokenType(str, Enum):
    """The type of credential token.

    :cvar oauth2_authorization_code: OAuth2 authorization code. Used to exchange for an access token (and optionally, refresh token).
                                     This should be provided once the client receives the OAuth2 callback.
    :cvar access_token: An opaque OAuth2 / JWT access token. Often what is vended as a "API key".

    """

    oauth2_authorization_code = "oauth2_authorization_code"
    access_token = "access_token"


class CredentialListItem(BaseModel):
    credential_id: str
    provider_id: str
    token_type: CredentialTokenType
    expires_at: datetime = Field(description="The time at which the credential expires. In UTC.")


@runtime_checkable
class Credentials(Protocol):
    """
    Create, update and delete ephemeral provider-specific credentials.

    Each provider may need optional authentication. This might be a persistent API key, a short-lived OAuth2
    access token or a refreshable OAuth2 token. There is a single credential for each provider instance.

    Credentials are ephemeral -- they may be purged after the specified TTL.

    Credentials are associated with the logged in user. If no user is logged in, the credentials
    are associated with the anonymous user.

    It is recommended to store these credentials using Envelope Encryption. The storage could
    be a regular KVStore, but you should use a secure Key Management Service for encrypting
    and decrypting.
    """

    @webmethod(route="/credentials", method="POST")
    async def create_credential(
        self,
        provider_id: str,
        token_type: CredentialTokenType,
        token: str,
        nonce: str | None = None,
        ttl_seconds: int = 3600,
    ) -> str:
        """Create a new set of credentials for a given provider.

        :param provider_id: The ID of the provider to create credentials for.
        :param token_type: The type of token to create. This is provided in the API to serve as lightweight
                           documentation / metadata for the token.
        :param token: The token itself.
        :param nonce: The nonce is required when the token type is oauth2_authorization_code.
        :param ttl_seconds: The time to live for the credential in seconds. Defaults to 3600 seconds.
                            When token_type is oauth2_authorization_code, the TTL is ignored and is obtained
                            from the provider when exchanging the code for an access token.
        :returns: The ID of the created credential.
        """
        ...

    @webmethod(route="/credentials/{credential_id}", method="DELETE")
    async def delete_credential(self, credential_id: str) -> None:
        """Delete a credential by its ID.

        :param credential_id: The ID of the credential to delete.
        """
        ...

    @webmethod(route="/credentials", method="GET")
    async def get_credentials(self) -> list[CredentialListItem]:
        """Get all credentials for the current user.

        :returns: A list of all credentials for the current user.
        """
        ...
