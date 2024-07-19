from models.llama3.datatypes import URL
from pydantic import BaseModel 


class Checkpoint(BaseModel):
    iters: int
    path: URL
