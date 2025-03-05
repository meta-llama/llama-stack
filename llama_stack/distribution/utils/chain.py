# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import logging
from itertools import pairwise
from typing import List

from llama_stack.apis.preprocessing import (
    Preprocessing,
    PreprocessingDataType,
    PreprocessorChain,
    PreprocessorInput,
    PreprocessorResponse,
)

log = logging.getLogger(__name__)


def validate_chain(chain_impls: List[Preprocessing], is_rag_chain: bool) -> bool:
    if len(chain_impls) == 0:
        log.error("Empty preprocessing chain was provided")
        return False

    if is_rag_chain and PreprocessingDataType.chunks not in chain_impls[-1].output_types:
        log.error(
            f"RAG preprocessing chain must end with a chunk-producing preprocessor, but the last preprocessor in the provided chain only supports {chain_impls[-1].output_types}"
        )
        return False

    for current_preprocessor, next_preprocessor in pairwise(chain_impls):
        current_output_types = current_preprocessor.output_types
        next_input_types = next_preprocessor.input_types

        if len(list(set(current_output_types) & set(next_input_types))) == 0:
            log.error(
                f"Incompatible input ({current_output_types}) and output({next_input_types}) preprocessor data types"
            )
            return False

    return True


async def execute_preprocessor_chain(
    preprocessor_chain: PreprocessorChain,
    preprocessor_chain_impls: List[Preprocessing],
    preprocessor_inputs: List[PreprocessorInput],
    is_rag_chain: bool,
) -> PreprocessorResponse:
    if not validate_chain(preprocessor_chain_impls, is_rag_chain):
        return PreprocessorResponse(status=False, results=[])

    current_inputs = preprocessor_inputs
    current_outputs = []

    # TODO: replace with a parallel implementation
    for i, current_params in enumerate(preprocessor_chain):
        current_impl = preprocessor_chain_impls[i]
        response = await current_impl.preprocess(
            preprocessor_id=current_params.preprocessor_id,
            preprocessor_inputs=current_inputs,
            options=current_params.options,
        )
        if not response.status:
            log.error(f"Preprocessor {current_params.preprocessor_id} returned an error")
            return PreprocessorResponse(status=False, results=[])
        current_outputs = response.results
        current_inputs = current_outputs

    return PreprocessorResponse(status=True, results=current_outputs)
