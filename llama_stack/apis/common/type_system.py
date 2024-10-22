# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict, List, Literal, Union

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
    items: "ParamType"


class ObjectType(BaseModel):
    type: Literal["object"] = "object"
    properties: Dict[str, "ParamType"] = Field(default_factory=dict)


class JsonType(BaseModel):
    type: Literal["json"] = "json"


class UnionType(BaseModel):
    type: Literal["union"] = "union"
    options: List["ParamType"] = Field(default_factory=list)


class CustomType(BaseModel):
    type: Literal["custom"] = "custom"
    validator_class: str


ParamType = Annotated[
    Union[
        StringType,
        NumberType,
        BooleanType,
        ArrayType,
        ObjectType,
        JsonType,
        UnionType,
        CustomType,
    ],
    Field(discriminator="type"),
]

ArrayType.model_rebuild()
ObjectType.model_rebuild()
UnionType.model_rebuild()
