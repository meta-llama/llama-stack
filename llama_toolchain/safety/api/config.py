from typing import List, Optional

from pydantic import BaseModel


class LlamaGuardShieldConfig(BaseModel):
    model_dir: str
    excluded_categories: List[str]
    disable_input_check: bool = False
    disable_output_check: bool = False


class PromptGuardShieldConfig(BaseModel):
    model_dir: str


class SafetyConfig(BaseModel):
    llama_guard_shield: Optional[LlamaGuardShieldConfig] = None
    prompt_guard_shield: Optional[PromptGuardShieldConfig] = None
