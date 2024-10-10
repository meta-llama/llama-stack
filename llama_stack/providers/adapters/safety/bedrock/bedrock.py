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


class BedrockSafetyAdapter(Safety, ShieldsProtocolPrivate):
    def __init__(self, config: BedrockSafetyConfig) -> None:
        if not config.aws_profile:
            raise ValueError(f"Missing boto_client aws_profile in model info::{config}")
        self.config = config
        self.registered_shields = []

    async def initialize(self) -> None:
        try:
            print(f"initializing with profile --- > {self.config}")
            self.boto_client = boto3.Session(
                profile_name=self.config.aws_profile
            ).client("bedrock-runtime")
        except Exception as e:
            raise RuntimeError("Error initializing BedrockSafetyAdapter") from e

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: ShieldDef) -> None:
        raise ValueError("Registering dynamic shields is not supported")

    async def list_shields(self) -> List[ShieldDef]:
        raise NotImplementedError(
            """
            `list_shields` not implemented; this should read all guardrails from
            bedrock and populate guardrailId and guardrailVersion in the ShieldDef.
        """
        )

    async def run_shield(
        self, shield_type: str, messages: List[Message], params: Dict[str, Any] = None
    ) -> RunShieldResponse:
        shield_def = await self.shield_store.get_shield(shield_type)
        if not shield_def:
            raise ValueError(f"Unknown shield {shield_type}")

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

        response = self.boto_client.apply_guardrail(
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

            return SafetyViolation(
                user_message=user_message,
                violation_level=ViolationLevel.ERROR,
                metadata=metadata,
            )

        return None
