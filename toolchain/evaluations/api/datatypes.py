from enum import Enum

from pydantic import BaseModel


class TextGenerationMetric(Enum):
    perplexity = "perplexity"
    rouge = "rouge"
    bleu = "bleu"


class QuestionAnsweringMetric(Enum):
    em = "em"
    f1 = "f1"


class SummarizationMetric(Enum):
    rouge = "rouge"
    bleu = "bleu"


class EvaluationJob(BaseModel):

    job_uuid: str


class EvaluationJobLogStream(BaseModel):

    job_uuid: str
