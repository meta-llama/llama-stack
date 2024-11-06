# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List

from llama_models.llama3.api.datatypes import interleaved_text_media_as_str, Message
from termcolor import cprint

from .config import CodeScannerConfig

from llama_stack.apis.safety import *  # noqa: F403


class MetaReferenceCodeScannerSafetyImpl(Safety):
    def __init__(self, config: CodeScannerConfig, deps) -> None:
        self.config = config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: ShieldDef) -> None:
        if shield.shield_type != ShieldType.code_scanner.value:
            raise ValueError(f"Unsupported safety shield type: {shield.shield_type}")

    async def run_shield(
        self,
        shield_type: str,
        messages: List[Message],
        params: Dict[str, Any] = None,
    ) -> RunShieldResponse:
        shield_def = await self.shield_store.get_shield(shield_type)
        if not shield_def:
            raise ValueError(f"Unknown shield {shield_type}")

        from codeshield.cs import CodeShield

        text = "\n".join([interleaved_text_media_as_str(m.content) for m in messages])
        cprint(f"Running CodeScannerShield on {text[50:]}", color="magenta")
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
