# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
from typing import Any, Dict, List

from llama_models.llama3.api.datatypes import interleaved_text_media_as_str, Message

from .config import CodeScannerConfig

from llama_stack.apis.safety import *  # noqa: F403

log = logging.getLogger(__name__)
ALLOWED_CODE_SCANNER_MODEL_IDS = [
    "CodeScanner",
    "CodeShield",
]


class MetaReferenceCodeScannerSafetyImpl(Safety):
    def __init__(self, config: CodeScannerConfig, deps) -> None:
        self.config = config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        if shield.provider_resource_id not in ALLOWED_CODE_SCANNER_MODEL_IDS:
            raise ValueError(
                f"Unsupported Code Scanner ID: {shield.provider_resource_id}. Allowed IDs: {ALLOWED_CODE_SCANNER_MODEL_IDS}"
            )

    async def run_shield(
        self,
        shield_id: str,
        messages: List[Message],
        params: Dict[str, Any] = None,
    ) -> RunShieldResponse:
        shield = await self.shield_store.get_shield(shield_id)
        if not shield:
            raise ValueError(f"Shield {shield_id} not found")

        from codeshield.cs import CodeShield

        text = "\n".join([interleaved_text_media_as_str(m.content) for m in messages])
        log.info(f"Running CodeScannerShield on {text[50:]}")
        result = await CodeShield.scan_code(text)

        violation = None
        if result.is_insecure:
            violation = SafetyViolation(
                violation_level=(ViolationLevel.ERROR),
                user_message="Sorry, I found security concerns in the code.",
                metadata={
                    "violation_type": ",".join(
                        [issue.pattern_id for issue in result.issues_found]
                    )
                },
            )
        return RunShieldResponse(violation=violation)
