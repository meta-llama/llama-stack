import os
from llama_models.datatypes import Model

from .config_dirs import DEFAULT_CHECKPOINT_DIR


def model_local_dir(model: Model) -> str:
    return os.path.join(DEFAULT_CHECKPOINT_DIR, model.descriptor())
