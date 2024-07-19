from typing import List, Protocol, Union
from .datatypes import *  # noqa: F403

from pyopenapi import webmethod


@json_schema_type
class RewardScoringRequest(BaseModel):
    """Request to score a reward function. A list of prompts and a list of responses per prompt."""

    dialog_generations: List[DialogGenerations]
    model: RewardModel


@json_schema_type
class RewardScoringResponse(BaseModel):
    """Response from the reward scoring. Batch of (prompt, response, score) tuples that pass the threshold."""

    scored_generations: List[ScoredDialogGenerations]


class RewardScoring(Protocol):
    @webmethod(route="/reward_scoring/score")
    def post_score(
        self,
        request: RewardScoringRequest,
    ) -> Union[RewardScoringResponse]: ...
