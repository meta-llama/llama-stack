# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described found in the
# LICENSE file in the root directory of this source tree.

# supress warnings and spew of logs from hugging face
import transformers

from .base import (  # noqa: F401
    DummyShield,
    OnViolationAction,
    ShieldBase,
    ShieldResponse,
    TextShield,
)
from .code_scanner import CodeScannerShield  # noqa: F401
from .contrib.third_party_shield import ThirdPartyShield  # noqa: F401
from .llama_guard import LlamaGuardShield  # noqa: F401
from .prompt_guard import (  # noqa: F401
    InjectionShield,
    JailbreakShield,
    PromptGuardShield,
)
from .shield_runner import SafetyException, ShieldRunnerMixin  # noqa: F401

transformers.logging.set_verbosity_error()

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings("ignore")
