from typing import List

from pydantic import BaseModel

from strong_typing.schema import json_schema_type

from llama_models.llama3_1.api.datatypes import *  # noqa: F403


@json_schema_type
class ScoredMessage(BaseModel):
    message: Message
    score: float


@json_schema_type
class DialogGenerations(BaseModel):
    dialog: List[Message]
    sampled_generations: List[Message]


@json_schema_type
class ScoredDialogGenerations(BaseModel):
    dialog: List[Message]
    scored_generations: List[ScoredMessage]
