# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from llama_stack.apis.evals import *  # noqa: F403
from llama_stack.providers.impls.meta_reference.evals.processor import *  # noqa: F403

from ..registry import Registry

# TODO: decide whether we should group dataset+processor together via Tasks
GeneratorProcessorRegistry = Registry[BaseGeneratorProcessor]()

PROCESSOR_REGISTRY = {
    "mmlu": MMLUProcessor,
    "judge": JudgeProcessor,
}

for k, v in PROCESSOR_REGISTRY.items():
    GeneratorProcessorRegistry.register(k, v)
