# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.models.llama.sku_list import resolve_model


def test_resolve_by_descriptor():
    """Test normal resolution by model descriptor."""
    model = resolve_model("Llama-4-Scout-17B-16E-Instruct")
    assert model is not None
    assert model.core_model_id.value == "Llama-4-Scout-17B-16E-Instruct"


def test_resolve_by_huggingface_repo():
    """Test normal resolution by HuggingFace repo path."""
    model = resolve_model("meta-llama/Llama-4-Scout-17B-16E-Instruct")
    assert model is not None
    assert model.core_model_id.value == "Llama-4-Scout-17B-16E-Instruct"


def test_together_alias_resolves():
    """Test that Together-prefixed alias resolves via generic prefix stripping."""
    alias = "together/meta-llama/Llama-4-Scout-17B-16E-Instruct"
    model = resolve_model(alias)
    assert model is not None, f"Model should resolve for alias {alias}"
    assert model.core_model_id.value == "Llama-4-Scout-17B-16E-Instruct"


def test_groq_alias_resolves():
    """Test that Groq-prefixed alias resolves via generic prefix stripping."""
    alias = "groq/meta-llama/Llama-4-Scout-17B-16E-Instruct"
    model = resolve_model(alias)
    assert model is not None, f"Model should resolve for alias {alias}"
    assert model.core_model_id.value == "Llama-4-Scout-17B-16E-Instruct"


def test_unknown_model_returns_none():
    """Test that unknown model descriptors return None."""
    model = resolve_model("nonexistent-model")
    assert model is None


def test_unknown_provider_prefix_returns_none():
    """Test that unknown provider prefix with unknown model returns None."""
    model = resolve_model("unknown-provider/nonexistent-model")
    assert model is None


def test_empty_string_returns_none():
    """Test that empty string returns None."""
    model = resolve_model("")
    assert model is None


def test_slash_only_returns_none():
    """Test that just a slash returns None."""
    model = resolve_model("/")
    assert model is None


def test_multiple_slashes_handled():
    """Test that paths with multiple slashes are handled correctly."""
    # This should strip "provider/" and try "path/to/model"
    model = resolve_model("provider/path/to/model")
    assert model is None  # Should be None since "path/to/model" doesn't exist
