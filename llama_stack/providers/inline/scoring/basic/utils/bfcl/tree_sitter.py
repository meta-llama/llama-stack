# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Tree-sitter changes its API with unfortunate frequency. Modules that need it should
import it from here so that we can centrally manage things as necessary.
"""

# These currently work with tree-sitter 0.23.0
# NOTE: Don't import tree-sitter or any of the language modules in the main module
# because not all environments have them. Import lazily inside functions where needed.

import importlib
import typing

if typing.TYPE_CHECKING:
    import tree_sitter


def get_language(language: str) -> "tree_sitter.Language":
    import tree_sitter

    language_module_name = f"tree_sitter_{language}"
    try:
        language_module = importlib.import_module(language_module_name)
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Language {language} is not found. Please install the tree-sitter-{language} package."
        ) from exc
    return tree_sitter.Language(language_module.language())


def get_parser(language: str, **kwargs) -> "tree_sitter.Parser":
    import tree_sitter

    lang = get_language(language)
    return tree_sitter.Parser(lang, **kwargs)
