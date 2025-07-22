# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json

import httpx

from llama_stack.cli.subcommand import Subcommand
from llama_stack.cli.table import print_table


class ShieldList(Subcommand):
    """List available shields"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "list",
            prog="llama shield list",
            description="Show available shields",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_shield_list_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "--url",
            type=str,
            default="http://localhost:8321",
            help="URL of the Llama Stack server (default: http://localhost:8321)",
        )
        self.parser.add_argument(
            "--output-format",
            choices=["table", "json"],
            default="table",
            help="Output format (default: table)",
        )

    def _run_shield_list_cmd(self, args: argparse.Namespace) -> None:
        try:
            response = httpx.get(f"{args.url}/v1/shields")
            response.raise_for_status()

            data = response.json()
            shields = data.get("data", [])

            if args.output_format == "json":
                print(json.dumps(shields, indent=2))
                return

            if not shields:
                print("No shields found.")
                return

            headers = ["Shield ID", "Provider ID", "Provider Shield ID", "Parameters"]

            rows = []
            for shield in shields:
                params_str = ""
                if shield.get("params"):
                    params_str = json.dumps(shield["params"], separators=(",", ":"))
                    if len(params_str) > 50:
                        params_str = params_str[:47] + "..."

                rows.append(
                    [
                        shield.get("identifier", shield.get("shield_id", "-")),
                        shield.get("provider_id", "-"),
                        shield.get("provider_resource_id", shield.get("provider_shield_id", "-")),
                        params_str or "-",
                    ]
                )

            print_table(
                rows,
                headers,
                separate_rows=True,
            )

        except httpx.RequestError as e:
            print(f"Error connecting to Llama Stack server: {e}")
        except httpx.HTTPStatusError as e:
            print(f"HTTP error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            print(f"Error listing shields: {e}")
