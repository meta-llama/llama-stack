# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# Lazy import torch to avoid breaking text-only usage
try:
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


if TYPE_CHECKING or _TORCH_AVAILABLE:

    @dataclass
    class MaskedEmbedding:  # type: ignore
        embedding: "torch.Tensor"
        mask: "torch.Tensor"

    @dataclass
    class LLMInput:  # type: ignore
        """
        This is the input to the LLM from the "user" -- the user in this case views the
        Llama4 model holistically and does not care or know about its inner workings (e.g.,
        whether it has an encoder or if it is early fusion or not.)

        This is distinct from the "TransformerInput" class which is really the Llama4
        backbone operating on early fused modalities and producing text output
        """

        tokens: "torch.Tensor"
        # images are already pre-processed (resized, tiled, etc.)
        images: list["torch.Tensor"] | None = None

    @dataclass
    class TransformerInput:  # type: ignore
        """
        This is the "core" backbone transformer of the Llama4 model. Inputs for other modalities
        are expected to be "embedded" via encoders sitting before this layer in the model.
        """

        tokens: "torch.Tensor"
        # tokens_position defines the position of the tokens in each batch,
        # - when it is a tensor ([batch_size,]), it is the start position of the tokens in each batch
        # - when it is an int, the start position are the same for all batches
        tokens_position: "torch.Tensor" | int
        image_embedding: "MaskedEmbedding" | None = None

    @dataclass
    class LLMOutput:  # type: ignore
        logits: "torch.Tensor"

else:
    # Fallback stubs when torch unavailable
    @dataclass
    class MaskedEmbedding:  # type: ignore
        embedding: Any
        mask: Any

    @dataclass
    class LLMInput:  # type: ignore
        tokens: Any
        images: Any = None

    @dataclass
    class TransformerInput:  # type: ignore
        tokens: Any
        tokens_position: Any
        image_embedding: Any = None

    @dataclass
    class LLMOutput:  # type: ignore
        logits: Any


TransformerOutput = LLMOutput
