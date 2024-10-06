# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from termcolor import cprint

from .base import ShieldResponse, TextShield


class CodeScannerShield(TextShield):
    async def run_impl(self, text: str) -> ShieldResponse:
        from codeshield.cs import CodeShield

        cprint(f"Running CodeScannerShield on {text[50:]}", color="magenta")
        result = await CodeShield.scan_code(text)
        if result.is_insecure:
            return ShieldResponse(
                is_violation=True,
                violation_type=",".join(
                    [issue.pattern_id for issue in result.issues_found]
                ),
                violation_return_message="Sorry, I found security concerns in the code.",
            )
        else:
            return ShieldResponse(is_violation=False)
