# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Literal, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated


class StringType(BaseModel):
    type: Literal["string"] = "string"


class NumberType(BaseModel):
    type: Literal["number"] = "number"


class BooleanType(BaseModel):
    type: Literal["boolean"] = "boolean"


class ArrayType(BaseModel):
    type: Literal["array"] = "array"


class ObjectType(BaseModel):
    type: Literal["object"] = "object"


class JsonType(BaseModel):
    type: Literal["json"] = "json"


class UnionType(BaseModel):
    type: Literal["union"] = "union"


class ChatCompletionInputType(BaseModel):
    # expects List[Message] for messages
    type: Literal["chat_completion_input"] = "chat_completion_input"


class CompletionInputType(BaseModel):
    # expects InterleavedTextMedia for content
    type: Literal["completion_input"] = "completion_input"


class AgentTurnInputType(BaseModel):
    # expects List[Message] for messages (may also include attachments?)
    type: Literal["agent_turn_input"] = "agent_turn_input"


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

# TODO: recursive definition of ParamType in these containers
# will cause infinite recursion in OpenAPI generation script
# since we are going with ChatCompletionInputType and CompletionInputType
# we don't need to worry about ArrayType/ObjectType/UnionType for now
# ArrayType.model_rebuild()
# ObjectType.model_rebuild()
# UnionType.model_rebuild()


# class CustomType(BaseModel):
#     type: Literal["custom"] = "custom"
#     validator_class: str
