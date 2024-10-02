# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_stack.cli.subcommand import Subcommand


class StackListProviders(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "list-providers",
            prog="llama stack list-providers",
            description="Show available Llama Stack Providers for an API",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_providers_list_cmd)

    def _add_arguments(self):
        from llama_stack.distribution.datatypes import Api

        api_values = [a.value for a in Api]
        self.parser.add_argument(
            "api",
            type=str,
            choices=api_values,
            help="API to list providers for (one of: {})".format(api_values),
        )

    def _run_providers_list_cmd(self, args: argparse.Namespace) -> None:
        from llama_stack.cli.table import print_table
        from llama_stack.distribution.distribution import Api, get_provider_registry

        all_providers = get_provider_registry()
        providers_for_api = all_providers[Api(args.api)]

        # eventually, this should query a registry at llama.meta.com/llamastack/distributions
        headers = [
            "Provider Type",
            "PIP Package Dependencies",
        ]

        rows = []
        for spec in providers_for_api.values():
            if spec.provider_type == "sample":
                continue
            rows.append(
                [
                    spec.provider_type,
                    ",".join(spec.pip_packages),
                ]
            )
        print_table(
            rows,
            headers,
            separate_rows=True,
        )
