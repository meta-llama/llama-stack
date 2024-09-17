# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from codeshield.cs import CodeShield
from termcolor import cprint

from .base import ShieldResponse, TextShield
from llama_stack.apis.safety import *  # noqa: F403


class CodeScannerShield(TextShield):
    def get_shield_type(self) -> ShieldType:
        return BuiltinShield.code_scanner_guard

    async def run_impl(self, text: str) -> ShieldResponse:
        cprint(f"Running CodeScannerShield on {text[50:]}", color="magenta")
        result = await CodeShield.scan_code(text)
        if result.is_insecure:
            return ShieldResponse(
                shield_type=BuiltinShield.code_scanner_guard,
                is_violation=True,
                violation_type=",".join(
                    [issue.pattern_id for issue in result.issues_found]
                ),
                violation_return_message="Sorry, I found security concerns in the code.",
            )
        else:
            return ShieldResponse(
                shield_type=BuiltinShield.code_scanner_guard, is_violation=False
            )
