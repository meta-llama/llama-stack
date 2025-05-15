# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_stack.distribution.resolver import resolve_storage_dir


class DummyConfig:
    pass


def test_storage_dir_cli(monkeypatch):
    config = DummyConfig()
    config.storage_dir = "/cli/dir"
    monkeypatch.delenv("LLAMA_STACK_STORAGE_DIR", raising=False)
    result = resolve_storage_dir(config, "distro")
    assert result == Path("/cli/dir")


def test_storage_dir_env(monkeypatch):
    config = DummyConfig()
    if hasattr(config, "storage_dir"):
        delattr(config, "storage_dir")
    monkeypatch.setenv("LLAMA_STACK_STORAGE_DIR", "/env/dir")
    result = resolve_storage_dir(config, "distro")
    assert result == Path("/env/dir")


def test_storage_dir_fallback(monkeypatch):
    # Mock the DISTRIBS_BASE_DIR
    monkeypatch.setattr("llama_stack.distribution.utils.config_dirs.DISTRIBS_BASE_DIR", Path("/mock/distribs"))

    config = DummyConfig()
    if hasattr(config, "storage_dir"):
        delattr(config, "storage_dir")
    monkeypatch.delenv("LLAMA_STACK_STORAGE_DIR", raising=False)

    result = resolve_storage_dir(config, "distro")
    assert result == Path("/mock/distribs/distro")
