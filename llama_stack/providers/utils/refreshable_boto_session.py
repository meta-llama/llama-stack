# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from datetime import datetime
from time import time

import pytz
from boto3 import Session
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session


class RefreshableBotoSession:
    """
    Boto Helper class which lets us create a refreshable session so that we can cache the client or resource.

    Usage
    -----
    session = RefreshableBotoSession().refreshable_session()

    client = session.client("bedrock-runtime") # we now can cache this client object without worrying about expiring credentials
    """

    def __init__(
        self,
        region_name: str = None,
        profile_name: str = None,
        session_ttl: int = 30000,
    ):
        """
        Initialize `RefreshableBotoSession`

        Parameters
        ----------
        region_name : str (optional)
            Default region when creating a new connection. Will check AWS_REGION or AWS_DEFAULT_REGION env vars if not provided.

        profile_name : str (optional)
            The name of a profile to use. Will check environment variables before using profile.
        """
        # Check environment variables for region
        self.region_name = (
            region_name
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
        )
        self.profile_name = profile_name
        self.session_ttl = session_ttl

    def __get_session_credentials(self):
        """
        Get session credentials from environment variables or session
        """
        # Check for credentials in environment variables first
        if all(
            key in os.environ for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        ):
            expiry_time = (
                os.environ.get("EXPIRY_TIME")
                or datetime.fromtimestamp(time() + self.session_ttl)
                .replace(tzinfo=pytz.utc)
                .isoformat()
            )
            credentials = {
                "access_key": os.environ["AWS_ACCESS_KEY_ID"],
                "secret_key": os.environ["AWS_SECRET_ACCESS_KEY"],
                "token": os.environ.get("AWS_SESSION_TOKEN"),  # Optional
                "expiry_time": expiry_time,
            }
            return credentials

        # Fall back to profile-based credentials
        session = Session(region_name=self.region_name, profile_name=self.profile_name)

        session_credentials = session.get_credentials().get_frozen_credentials()
        credentials = {
            "access_key": session_credentials.access_key,
            "secret_key": session_credentials.secret_key,
            "token": session_credentials.token,
            "expiry_time": datetime.fromtimestamp(time() + self.session_ttl)
            .replace(tzinfo=pytz.utc)
            .isoformat(),
        }

        return credentials

    def refreshable_session(self) -> Session:
        """
        Get refreshable boto3 session.
        """
        # Get refreshable credentials
        refreshable_credentials = RefreshableCredentials.create_from_metadata(
            metadata=self.__get_session_credentials(),
            refresh_using=self.__get_session_credentials,
            method="sts-assume-role",
        )

        # attach refreshable credentials current session
        session = get_session()
        session._credentials = refreshable_credentials
        session.set_config_variable("region", self.region_name)
        autorefresh_session = Session(botocore_session=session)

        return autorefresh_session
