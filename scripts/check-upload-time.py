# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re
import sys


def check_upload_time_regex(file_content):
    sdist_pattern = re.compile(r'sdist\s*=\s*\{[^}]*upload-time\s*=\s*".*?"[^}]*\}', re.DOTALL)
    sdist_found = bool(sdist_pattern.search(file_content))
    return sdist_found


def main():
    try:
        with open("uv.lock", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading uv.lock: {e}")
        sys.exit(1)

    if not check_upload_time_regex(content):
        print(
            "It looks like you're using an old version of uv. "
            "The lock file does not include upload-time fields for the packages. "
            "Please update uv and regenerate the lock file"
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
