# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.agentic_system.api import *  # noqa: F403
from llama_stack.dataset.api import *  # noqa: F403
from llama_stack.evaluations.api import *  # noqa: F403
from llama_stack.inference.api import *  # noqa: F403
from llama_stack.batch_inference.api import *  # noqa: F403
from llama_stack.memory.api import *  # noqa: F403
from llama_stack.telemetry.api import *  # noqa: F403
from llama_stack.post_training.api import *  # noqa: F403
from llama_stack.reward_scoring.api import *  # noqa: F403
from llama_stack.synthetic_data_generation.api import *  # noqa: F403
from llama_stack.safety.api import *  # noqa: F403


class LlamaStack(
    Inference,
    BatchInference,
    AgenticSystem,
    RewardScoring,
    Safety,
    SyntheticDataGeneration,
    Datasets,
    Telemetry,
    PostTraining,
    Memory,
    Evaluations,
):
    pass
