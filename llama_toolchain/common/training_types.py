from llama_models.llama3_1.api.datatypes import URL
from pydantic import BaseModel 


class Checkpoint(BaseModel):
    iters: int
    path: URL
