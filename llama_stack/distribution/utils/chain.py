# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import logging
from itertools import pairwise
from typing import List

from llama_stack.apis.preprocessing import (
    PreprocessingDataElement,
    PreprocessorChain,
    PreprocessorResponse,
)
from llama_stack.providers.datatypes import PreprocessorsProtocolPrivate

log = logging.getLogger(__name__)


def validate_chain(chain_impls: List[PreprocessorsProtocolPrivate]) -> bool:
    if len(chain_impls) == 0:
        log.error("Empty preprocessing chain was provided")
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
    preprocessor_chain_impls: List[PreprocessorsProtocolPrivate],
    preprocessor_inputs: List[PreprocessingDataElement],
) -> PreprocessorResponse:
    if not validate_chain(preprocessor_chain_impls):
        return PreprocessorResponse(success=False, output_data_type=None, results=[])

    current_inputs = preprocessor_inputs
    current_outputs: List[PreprocessingDataElement] | None = []
    current_result_type = None

    # TODO: replace with a parallel implementation
    for i, current_params in enumerate(preprocessor_chain):
        current_impl = preprocessor_chain_impls[i]
        response = await current_impl.do_preprocess(
            preprocessor_id=current_params.preprocessor_id,
            preprocessor_inputs=current_inputs,
            options=current_params.options,
        )
        if not response.success:
            log.error(f"Preprocessor {current_params.preprocessor_id} returned an error")
            return PreprocessorResponse(success=False, output_data_type=response.output_data_type, results=[])
        current_outputs = response.results
        if current_outputs is None:
            log.error(f"Preprocessor {current_params.preprocessor_id} returned invalid results")
            return PreprocessorResponse(success=False, output_data_type=response.output_data_type, results=[])
        current_inputs = current_outputs
        current_result_type = response.output_data_type

    return PreprocessorResponse(success=True, output_data_type=current_result_type, results=current_outputs)
