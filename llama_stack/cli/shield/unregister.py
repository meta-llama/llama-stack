# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import sys

import httpx

from llama_stack.cli.subcommand import Subcommand


class ShieldUnregister(Subcommand):
    """Unregister a shield"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "unregister",
            prog="llama shield unregister",
            description="Unregister a shield",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_shield_unregister_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "shield_id",
            type=str,
            help="The identifier of the shield to unregister",
        )
        self.parser.add_argument(
            "--url",
            type=str,
            default="http://localhost:8321",
            help="URL of the Llama Stack server (default: http://localhost:8321)",
        )
        self.parser.add_argument(
            "--force",
            action="store_true",
            help="Force unregister without confirmation",
        )

    def _run_shield_unregister_cmd(self, args: argparse.Namespace) -> None:
        try:
            try:
                response = httpx.get(f"{args.url}/v1/shields/{args.shield_id}")
                response.raise_for_status()
                shield = response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400 or e.response.status_code == 404:
                    print(f"Shield '{args.shield_id}' not found.")
                    sys.exit(1)
                else:
                    raise

            if not args.force:
                shield_id = shield.get("identifier", shield.get("shield_id", args.shield_id))
                provider_id = shield.get("provider_id", "<Not Set>")
                provider_shield_id = shield.get("provider_resource_id", shield.get("provider_shield_id", "<Not Set>"))

                print("Shield to unregister:")
                print(f"  - Shield ID: {shield_id}")
                print(f"  - Provider ID: {provider_id}")
                print(f"  - Provider Shield ID: {provider_shield_id}")

                response_input = input(f"\nAre you sure you want to unregister shield '{args.shield_id}'? (y/N): ")
                if response_input.lower() not in ["y", "yes"]:
                    print("Unregister cancelled.")
                    return

            response = httpx.delete(f"{args.url}/v1/shields/{args.shield_id}")
            response.raise_for_status()

            print(f"Shield '{args.shield_id}' unregistered successfully!")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400 or e.response.status_code == 404:
                print(f"Shield '{args.shield_id}' not found.")
            else:
                print(f"HTTP error {e.response.status_code}: {e.response.text}")
            sys.exit(1)
        except httpx.RequestError as e:
            print(f"Error connecting to Llama Stack server: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error unregistering shield: {e}")
            sys.exit(1)
