# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from pydantic import BaseModel, Field

from llama_stack.apis.credentials import CredentialListItem, CredentialTokenType
from llama_stack.apis.credentials import Credentials as CredentialsAPI
from llama_stack.distribution.request_headers import get_logged_in_user
from llama_stack.log import get_logger
from llama_stack.providers.utils.kvstore import KVStore, KVStoreConfig, kvstore_impl

from .datatypes import Api

logger = get_logger(__name__, category="core")


class AuthenticationRequiredError(Exception):
    pass


class ProviderCredential(BaseModel):
    credential_id: str
    provider_id: str
    token_type: CredentialTokenType
    access_token: str
    expires_at: datetime = Field(description="The time at which the credential expires. In UTC.")
    refresh_token: str | None = None


class CredentialsStore(Protocol):
    """This is a private protocol used by the distribution and providers to operate on credentials."""

    async def read_decrypted_credential(self, provider_id: str) -> str | None: ...


class DistributionCredentialsConfig(BaseModel):
    # TODO: a kvstore isn't the right primitive because we need to look up
    # by both `credential_id` (for delete) and (user, provider_id) for fast look ups
    kvstore: KVStoreConfig


def get_principal() -> str:
    logged_in_user = get_logged_in_user()
    if not logged_in_user:
        # unauth stack, all users have access to this credential
        principal = "*"
    else:
        principal = logged_in_user
    return principal


class DistributionCredentialsImpl(CredentialsAPI, CredentialsStore):
    def __init__(self, config: DistributionCredentialsConfig, deps: dict[Api, Any]):
        self.config = config
        self.deps = deps
        self.store: KVStore | None = None

    async def initialize(self) -> None:
        self.store = await kvstore_impl(self.config.kvstore)

    async def shutdown(self) -> None:
        pass

    async def get_credentials(self) -> list[CredentialListItem]:
        principal = get_principal()
        assert self.store is not None

        credentials = []
        start = f"principal:{principal}/"
        end = f"principal:{principal}/\xff"
        for value in await self.store.values_in_range(start, end):
            if not value:
                continue
            credential = ProviderCredential(**json.loads(value))
            credentials.append(
                CredentialListItem(
                    credential_id=credential.credential_id,
                    provider_id=credential.provider_id,
                    token_type=credential.token_type,
                    expires_at=credential.expires_at,
                )
            )
        return credentials

    async def create_credential(
        self,
        provider_id: str,
        token_type: CredentialTokenType,
        token: str,
        nonce: str | None = None,
        ttl_seconds: int = 3600,
    ) -> str:
        if token_type == CredentialTokenType.oauth2_authorization_code:
            # TODO: we need to exchange the authorization code for an access token
            # and store { access_token, refresh_token, expires_at }
            raise NotImplementedError("OAuth2 authorization code is not supported yet")

        principal = get_principal()

        # check that provider_id is registered
        run_config = self.deps[Api.inspect].run_config

        # TODO: we should make provider_ids unique across all APIs which is not enforced yet
        provider_ids = [p.provider_id for p in run_config.providers.values()]
        if provider_id not in provider_ids:
            raise ValueError(f"Provider {provider_id} is not registered")

        credential_id = str(uuid.uuid4())
        credential = ProviderCredential(
            credential_id=credential_id,
            provider_id=provider_id,
            token_type=token_type,
            access_token=token,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
            refresh_token=None,
        )
        await self.store_credential(principal, credential)
        return credential_id

    async def delete_credential(self, credential_id: str) -> None:
        principal = get_principal()
        assert self.store is not None

        credentials = await self.get_credentials()
        for credential in credentials:
            if credential.credential_id == credential_id:
                await self.store.delete(make_credential_key(principal, credential.provider_id))
                return
        raise ValueError(f"Credential {credential_id} not found")

    async def store_credential(self, principal: str, credential: ProviderCredential) -> None:
        # TODO: encrypt
        key = make_credential_key(principal, credential.provider_id)
        assert self.store is not None
        await self.store.set(key, credential.model_dump_json())

    async def read_decrypted_credential(self, provider_id: str) -> str | None:
        principal = get_principal()

        key = make_credential_key(principal, provider_id)
        assert self.store is not None
        value = await self.store.get(key)
        if not value:
            return None
        credential = ProviderCredential(**json.loads(value))
        if credential.expires_at < datetime.now(timezone.utc):
            logger.info(f"Credential {credential.credential_id} for {provider_id} has expired")
            return None
        return credential.access_token


def make_credential_key(principal: str, provider_id: str) -> str:
    return f"principal:{principal}/provider:{provider_id}"
