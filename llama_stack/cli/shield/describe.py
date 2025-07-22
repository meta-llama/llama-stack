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


class ShieldDescribe(Subcommand):
    """Show details about a shield"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "describe",
            prog="llama shield describe",
            description="Show details about a shield",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_shield_describe_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "shield_id",
            type=str,
            help="The identifier of the shield to describe",
        )
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

    def _run_shield_describe_cmd(self, args: argparse.Namespace) -> None:
        try:
            response = httpx.get(f"{args.url}/v1/shields/{args.shield_id}")
            response.raise_for_status()

            shield = response.json()

            if args.output_format == "json":
                print(json.dumps(shield, indent=2))
                return

            headers = ["Property", "Value"]

            shield_id = shield.get("identifier", shield.get("shield_id", args.shield_id))
            provider_id = shield.get("provider_id", "<Not Set>")
            provider_shield_id = shield.get("provider_resource_id", shield.get("provider_shield_id", "<Not Set>"))
            resource_type = shield.get("type", "shield")

            rows = [
                ("Shield ID", shield_id),
                ("Provider ID", provider_id),
                ("Provider Shield ID", provider_shield_id),
                ("Resource Type", resource_type),
            ]

            if shield.get("params"):
                rows.append(("Parameters", json.dumps(shield["params"], indent=2)))
            else:
                rows.append(("Parameters", "<None>"))

            print_table(
                rows,
                headers,
                separate_rows=True,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400 or e.response.status_code == 404:
                print(f"Shield '{args.shield_id}' not found.")
            else:
                print(f"HTTP error {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            print(f"Error connecting to Llama Stack server: {e}")
        except Exception as e:
            print(f"Error describing shield: {e}")
