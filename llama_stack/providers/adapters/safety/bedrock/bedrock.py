# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import logging

from typing import Any, Dict, List

import boto3

from llama_stack.apis.safety import *  # noqa
from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.providers.datatypes import ShieldsProtocolPrivate

from .config import BedrockSafetyConfig


logger = logging.getLogger(__name__)


BEDROCK_SUPPORTED_SHIELDS = [
    ShieldType.generic_content_shield.value,
]


def _create_bedrock_client(config: BedrockSafetyConfig, name: str):
    session_args = {
        "aws_access_key_id": config.aws_access_key_id,
        "aws_secret_access_key": config.aws_secret_access_key,
        "aws_session_token": config.aws_session_token,
        "region_name": config.region_name,
        "profile_name": config.profile_name,
    }

    # Remove None values
    session_args = {k: v for k, v in session_args.items() if v is not None}

    boto3_session = boto3.session.Session(**session_args)
    return boto3_session.client(name)


class BedrockSafetyAdapter(Safety, ShieldsProtocolPrivate):
    def __init__(self, config: BedrockSafetyConfig) -> None:
        self.config = config
        self.registered_shields = []

    async def initialize(self) -> None:
        try:
            self.bedrock_runtime_client = _create_bedrock_client(
                self.config, "bedrock-runtime"
            )
            self.bedrock_client = _create_bedrock_client(self.config, "bedrock")
        except Exception as e:
            raise RuntimeError("Error initializing BedrockSafetyAdapter") from e

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: ShieldDef) -> None:
        raise ValueError("Registering dynamic shields is not supported")

    async def list_shields(self) -> List[ShieldDef]:
        response = self.bedrock_client.list_guardrails()
        shields = []
        for guardrail in response["guardrails"]:
            # populate the shield def with the guardrail id and version
            shield_def = ShieldDef(
                identifier=guardrail["id"],
                shield_type=ShieldType.generic_content_shield.value,
                params={
                    "guardrailIdentifier": guardrail["id"],
                    "guardrailVersion": guardrail["version"],
                },
            )
            self.registered_shields.append(shield_def)
            shields.append(shield_def)
        return shields

    async def run_shield(
        self, identifier: str, messages: List[Message], params: Dict[str, Any] = None
    ) -> RunShieldResponse:
        shield_def = await self.shield_store.get_shield(identifier)
        if not shield_def:
            raise ValueError(f"Unknown shield {identifier}")

        """This is the implementation for the bedrock guardrails. The input to the guardrails is to be of this format
        ```content = [
            {
                "text": {
                    "text": "Is the AB503 Product a better investment than the S&P 500?"
                }
            }
        ]```
        However the incoming messages are of this type UserMessage(content=....) coming from
        https://github.com/meta-llama/llama-models/blob/main/models/llama3/api/datatypes.py

        They contain content, role . For now we will extract the content and default the "qualifiers": ["query"]
        """

        shield_params = shield_def.params
        logger.debug(f"run_shield::{shield_params}::messages={messages}")

        # - convert the messages into format Bedrock expects
        content_messages = []
        for message in messages:
            content_messages.append({"text": {"text": message.content}})
        logger.debug(
            f"run_shield::final:messages::{json.dumps(content_messages, indent=2)}:"
        )

        response = self.bedrock_runtime_client.apply_guardrail(
            guardrailIdentifier=shield_params["guardrailIdentifier"],
            guardrailVersion=shield_params["guardrailVersion"],
            source="OUTPUT",  # or 'INPUT' depending on your use case
            content=content_messages,
        )
        if response["action"] == "GUARDRAIL_INTERVENED":
            user_message = ""
            metadata = {}
            for output in response["outputs"]:
                # guardrails returns a list - however for this implementation we will leverage the last values
                user_message = output["text"]
            for assessment in response["assessments"]:
                # guardrails returns a list - however for this implementation we will leverage the last values
                metadata = dict(assessment)

            return RunShieldResponse(
                violation=SafetyViolation(
                    user_message=user_message,
                    violation_level=ViolationLevel.ERROR,
                    metadata=metadata,
                )
            )

        return RunShieldResponse()
