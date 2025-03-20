# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Literal, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from llama_stack.schema_utils import json_schema_type, register_schema


@json_schema_type
class StringType(BaseModel):
    type: Literal["string"] = "string"


@json_schema_type
class NumberType(BaseModel):
    type: Literal["number"] = "number"


@json_schema_type
class BooleanType(BaseModel):
    type: Literal["boolean"] = "boolean"


@json_schema_type
class ArrayType(BaseModel):
    type: Literal["array"] = "array"


@json_schema_type
class ObjectType(BaseModel):
    type: Literal["object"] = "object"


@json_schema_type
class JsonType(BaseModel):
    type: Literal["json"] = "json"


@json_schema_type
class UnionType(BaseModel):
    type: Literal["union"] = "union"


@json_schema_type
class ChatCompletionInputType(BaseModel):
    # expects List[Message] for messages
    type: Literal["chat_completion_input"] = "chat_completion_input"


@json_schema_type
class CompletionInputType(BaseModel):
    # expects InterleavedTextMedia for content
    type: Literal["completion_input"] = "completion_input"


@json_schema_type
class AgentTurnInputType(BaseModel):
    # expects List[Message] for messages (may also include attachments?)
    type: Literal["agent_turn_input"] = "agent_turn_input"


@json_schema_type
class DialogType(BaseModel):
    # expects List[Message] for messages
    # this type semantically contains the output label whereas ChatCompletionInputType does not
    type: Literal["dialog"] = "dialog"


ParamType = Annotated[
    Union[
        StringType,
        NumberType,
        BooleanType,
        ArrayType,
        ObjectType,
        JsonType,
        UnionType,
        ChatCompletionInputType,
        CompletionInputType,
        AgentTurnInputType,
    ],
    Field(discriminator="type"),
]
register_schema(ParamType, name="ParamType")

"""
# TODO: recursive definition of ParamType in these containers
# will cause infinite recursion in OpenAPI generation script
# since we are going with ChatCompletionInputType and CompletionInputType
# we don't need to worry about ArrayType/ObjectType/UnionType for now
ArrayType.model_rebuild()
ObjectType.model_rebuild()
UnionType.model_rebuild()


class CustomType(BaseModel):
pylint: disable=syntax-error
    type: Literal["custom"] = "custom"
    validator_class: str
"""
