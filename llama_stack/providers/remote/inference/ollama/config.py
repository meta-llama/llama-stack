# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from llama_stack.distribution.datatypes import RemoteProviderConfig


DEFAULT_OLLAMA_PORT = 11434


class OllamaImplConfig(RemoteProviderConfig):
    port: int

    @classmethod
    def sample_run_config(
        cls, port_str: str = str(DEFAULT_OLLAMA_PORT)
    ) -> Dict[str, Any]:
        return {"port": port_str}
