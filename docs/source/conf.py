# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "llama-stack"
copyright = "2024, Meta"
author = "Meta"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_rtd_theme",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
    "sphinx_design",
]
myst_enable_extensions = ["colon_fence"]

html_theme = "sphinx_rtd_theme"
html_use_relative_paths = True

# html_theme = "sphinx_pdj_theme"
# html_theme_path = [sphinx_pdj_theme.get_html_theme_path()]

# html_theme = "pytorch_sphinx_theme"
# html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    # "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Copy button settings
copybutton_prompt_text = "$ "  # for bash prompts
copybutton_prompt_is_regexp = True
copybutton_remove_prompts = True
copybutton_line_continuation_character = "\\"

# Source suffix
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
html_theme_options = {
    "canonical_url": "https://github.com/meta-llama/llama-stack",
    # "style_nav_header_background": "#c3c9d4",
}

html_static_path = ["../_static"]
# html_logo = "../_static/llama-stack-logo.png"
html_style = "../_static/css/my_theme.css"
