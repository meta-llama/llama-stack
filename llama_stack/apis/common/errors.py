# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


class UnsupportedModelError(ValueError):
    """raised when model is not present in the list of supported models"""

    def __init__(self, model_name: str, supported_models_list: list[str]):
        message = f"'{model_name}' model is not supported. Supported models are: {', '.join(supported_models_list)}"
        super().__init__(message)


class ModelNotFoundError(ValueError):
    """raised when Llama Stack cannot find a referenced model"""

    def __init__(self, model_name: str) -> None:
        message = f"Model '{model_name}' not found. Use client.models.list() to list available models."
        super().__init__(message)
