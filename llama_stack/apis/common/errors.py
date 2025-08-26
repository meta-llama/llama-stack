# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Custom Llama Stack Exception classes should follow the following schema
#   1. All classes should inherit from an existing Built-In Exception class: https://docs.python.org/3/library/exceptions.html
#   2. All classes should have a custom error message with the goal of informing the Llama Stack user specifically
#   3. All classes should propogate the inherited __init__ function otherwise via 'super().__init__(message)'

from llama_stack.providers.utils.sqlstore.sqlstore import SqlStoreType


class ResourceNotFoundError(ValueError):
    """generic exception for a missing Llama Stack resource"""

    def __init__(self, resource_name: str, resource_type: str, client_list: str) -> None:
        message = (
            f"{resource_type} '{resource_name}' not found. Use '{client_list}' to list available {resource_type}s."
        )
        super().__init__(message)


class UnsupportedModelError(ValueError):
    """raised when model is not present in the list of supported models"""

    def __init__(self, model_name: str, supported_models_list: list[str]):
        message = f"'{model_name}' model is not supported. Supported models are: {', '.join(supported_models_list)}"
        super().__init__(message)


class UnsupportedSqlStoreError(ValueError):
    """raised when SQL store is not present in the list of supported SQL stores"""

    def __init__(self, sqlstore_type: str):
        message = (
            f"'{sqlstore_type}' SQL store is not supported. Supported SQL stores are: {', '.join(list(SqlStoreType))}"
        )
        super().__init__(message)


class ModelNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced model"""

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name, "Model", "client.models.list()")


class VectorStoreNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced vector store"""

    def __init__(self, vector_store_name: str) -> None:
        super().__init__(vector_store_name, "Vector Store", "client.vector_dbs.list()")


class DatasetNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced dataset"""

    def __init__(self, dataset_name: str) -> None:
        super().__init__(dataset_name, "Dataset", "client.datasets.list()")


class ToolGroupNotFoundError(ResourceNotFoundError):
    """raised when Llama Stack cannot find a referenced tool group"""

    def __init__(self, toolgroup_name: str) -> None:
        super().__init__(toolgroup_name, "Tool Group", "client.toolgroups.list()")


class SessionNotFoundError(ValueError):
    """raised when Llama Stack cannot find a referenced session or access is denied"""

    def __init__(self, session_name: str) -> None:
        message = f"Session '{session_name}' not found or access denied."
        super().__init__(message)


class ModelTypeError(TypeError):
    """raised when a model is present but not the correct type"""

    def __init__(self, model_name: str, model_type: str, expected_model_type: str) -> None:
        message = (
            f"Model '{model_name}' is of type '{model_type}' rather than the expected type '{expected_model_type}'"
        )
        super().__init__(message)


class ConflictError(ValueError):
    """raised when an operation cannot be performed due to a conflict with the current state"""

    def __init__(self, message: str) -> None:
        super().__init__(message)
