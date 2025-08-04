# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Custom Llama Stack Exception classes should follow the following schema
#   1. All classes should inherit from an existing Built-In Exception class: https://docs.python.org/3/library/exceptions.html
#   2. All classes should have a custom error message with the goal of informing the Llama Stack user specifically
#   3. All classes should propogate the inherited __init__ function otherwise via 'super().__init__(message)'


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


class VectorStoreNotFoundError(ValueError):
    """raised when Llama Stack cannot find a referenced vector store"""

    def __init__(self, vector_store_name: str) -> None:
        message = f"Vector store '{vector_store_name}' not found. Use client.vector_dbs.list() to list available vector stores."
        super().__init__(message)


class DatasetNotFoundError(ValueError):
    """raised when Llama Stack cannot find a referenced dataset"""

    def __init__(self, dataset_name: str) -> None:
        message = f"Dataset '{dataset_name}' not found. Use client.datasets.list() to list available datasets."
        super().__init__(message)


class ToolGroupNotFoundError(ValueError):
    """raised when Llama Stack cannot find a referenced tool group"""

    def __init__(self, toolgroup_name: str) -> None:
        message = (
            f"Tool group '{toolgroup_name}' not found. Use client.toolgroups.list() to list available tool groups."
        )
        super().__init__(message)
