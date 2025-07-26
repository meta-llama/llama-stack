# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json
import sys

import httpx

from llama_stack.cli.subcommand import Subcommand


class ShieldRegister(Subcommand):
    """Register a new shield"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "register",
            prog="llama shield register",
            description="Register a new shield",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_shield_register_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "shield_id",
            type=str,
            help="The identifier for the shield to register",
        )
        self.parser.add_argument(
            "--provider-id",
            type=str,
            help="The provider ID for the shield",
        )
        self.parser.add_argument(
            "--provider-shield-id",
            type=str,
            help="The provider-specific shield identifier",
        )
        self.parser.add_argument(
            "--params",
            type=str,
            help='Shield parameters as JSON string (e.g., \'{"key": "value"}\')',
        )
        self.parser.add_argument(
            "--url",
            type=str,
            default="http://localhost:8321",
            help="URL of the Llama Stack server (default: http://localhost:8321)",
        )

    def _run_shield_register_cmd(self, args: argparse.Namespace) -> None:
        try:
            params = None
            if args.params:
                try:
                    params = json.loads(args.params)
                except json.JSONDecodeError as e:
                    print(f"Error parsing parameters JSON: {e}")
                    sys.exit(1)

            payload = {
                "shield_id": args.shield_id,
            }

            if args.provider_id:
                payload["provider_id"] = args.provider_id
            if args.provider_shield_id:
                payload["provider_shield_id"] = args.provider_shield_id
            if params:
                payload["params"] = params

            response = httpx.post(f"{args.url}/v1/shields", json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()

            shield = response.json()

            print(f"   Shield '{shield.get('identifier', args.shield_id)}' registered successfully!")
            print(f"   Provider ID: {shield.get('provider_id', '<Not Set>')}")
            print(
                f"   Provider Shield ID: {shield.get('provider_resource_id', shield.get('provider_shield_id', '<Not Set>'))}"
            )
            if shield.get("params"):
                print(f"   Parameters: {json.dumps(shield['params'], indent=2)}")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400 and "already exists" in e.response.text:
                print(f"Shield '{args.shield_id}' already exists.")
            else:
                print(f"HTTP error {e.response.status_code}: {e.response.text}")
            sys.exit(1)
        except httpx.RequestError as e:
            print(f"Error connecting to Llama Stack server: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error registering shield: {e}")
            sys.exit(1)
