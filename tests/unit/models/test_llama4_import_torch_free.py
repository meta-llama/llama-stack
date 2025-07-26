# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Test that Llama-4 modules can be imported and used for text-only operations
even when torch is not available (addresses issue #2584).
"""

import builtins
import importlib
import sys

import pytest


def _block_torch(monkeypatch):
    """Block torch imports to simulate torch-free environment."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    sys.modules.pop("torch", None)  # forget any cached import
    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_llama4_chat_format_imports_without_torch(monkeypatch):
    """Test that llama4.chat_format can be imported when torch is unavailable."""
    _block_torch(monkeypatch)

    # This should NOT raise ImportError anymore
    chat_format_module = importlib.import_module("llama_stack.models.llama.llama4.chat_format")
    assert chat_format_module is not None


def test_llama4_text_decoding_works_without_torch(monkeypatch):
    """Test that text-only tool calling decoding works without torch."""
    _block_torch(monkeypatch)

    from llama_stack.models.llama.datatypes import StopReason
    from llama_stack.models.llama.llama4.chat_format import ChatFormat
    from llama_stack.models.llama.llama4.tokenizer import Tokenizer

    # Text-only operations should work fine
    formatter = ChatFormat(Tokenizer.get_instance())
    content = '[get_weather(location="SF")]<|eot|>'
    msg = formatter.decode_assistant_message_from_content(content, StopReason.end_of_turn)

    # Verify tool calling parsing works
    assert msg.tool_calls, "Tool call should be detected"
    tc = msg.tool_calls[0]
    assert tc.tool_name == "get_weather"
    assert tc.arguments == {"location": "SF"}


def test_llama4_vision_fails_gracefully_without_torch(monkeypatch):
    """Test that vision features raise clear error when torch unavailable."""
    _block_torch(monkeypatch)

    from llama_stack.models.llama.llama4.args import Size, VisionArgs
    from llama_stack.models.llama.llama4.chat_format import ChatFormat
    from llama_stack.models.llama.llama4.tokenizer import Tokenizer

    # Trying to use vision features should raise helpful error
    vision_args = VisionArgs(
        image_size=Size(height=448, width=448),
        patch_size=Size(height=14, width=14),
        dim=512,
        n_layers=6,
        n_heads=8,
        mlp_ratio=4.0,
        output_dim=4096,
        pixel_shuffle_ratio=2,
    )
    with pytest.raises(ImportError, match="vision features require.*torch.*Pillow"):
        ChatFormat(Tokenizer.get_instance(), vision_args=vision_args)
