#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import requests


def get_all_releases(token):
    url = "https://api.github.com/repos/meta-llama/llama-stack/releases"
    headers = {"Accept": "application/vnd.github.v3+json"}

    if token:
        headers["Authorization"] = f"token {token}"

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching releases: {response.status_code}, {response.text}")


def clean_release_body(body):
    """Remove '## All changes' sections from release notes."""
    lines = body.split("\n")
    cleaned_lines = []
    skip_mode = False

    for line in lines:
        if line.strip() in [
            "## All changes",
            "### What's Changed",
            "## What's Changed",
            "## New Contributors",
        ]:
            skip_mode = True
        elif skip_mode and line.startswith("##"):
            # Found a new section, stop skipping
            skip_mode = False
            cleaned_lines.append(line)
        elif not skip_mode:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def merge_release_notes(output_file, token=None):
    releases = get_all_releases(token)

    with open(output_file, "w", encoding="utf-8") as md_file:
        md_file.write("# Changelog\n\n")

        for release in releases:
            md_file.write(f"# {release['tag_name']}\n")
            md_file.write(f"Published on: {release['published_at']}\n\n")

            # Clean the release body to remove "## All changes" sections
            cleaned_body = clean_release_body(release["body"])
            md_file.write(f"{cleaned_body}\n\n")

            md_file.write("---\n\n")

    print(f"Merged release notes saved to {output_file}")


if __name__ == "__main__":
    OUTPUT_FILE = "CHANGELOG.md"
    TOKEN = os.getenv("GITHUB_TOKEN")
    merge_release_notes(OUTPUT_FILE, TOKEN)
