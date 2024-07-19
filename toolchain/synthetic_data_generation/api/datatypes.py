from enum import Enum


class FilteringFunction(Enum):
    """The type of filtering function."""

    none = "none"
    random = "random"
    top_k = "top_k"
    top_p = "top_p"
    top_k_top_p = "top_k_top_p"
    sigmoid = "sigmoid"
