# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


def get_provider_type(module: str) -> str:
    parts = module.split(".")
    if parts[0] != "llama_stack" or parts[1] != "providers":
        raise ValueError(f"Invalid module name <{module}>")
    if parts[2] == "inline" or parts[2] == "remote":
        return parts[2]
    else:
        raise ValueError(f"Invalid module name <{module}>")
