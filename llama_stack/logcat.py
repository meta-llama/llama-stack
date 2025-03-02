# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Category-based logging utility for llama-stack.

This module provides a wrapper over the standard Python logging module that supports
categorized logging with environment variable control.

Usage:
    from llama_stack import logcat
    logcat.info("server", "Starting up...")
    logcat.debug("inference", "Processing request...")

Environment variable:
    LLAMA_STACK_LOGGING: Semicolon-separated list of category=level pairs
    Example: "server=debug;inference=warning"
"""

import datetime
import logging
import os
from typing import Dict

# ANSI color codes for terminal output
COLORS = {
    "RESET": "\033[0m",
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
    "DIM": "\033[2m",  # Dimmed text
    "YELLOW_DIM": "\033[2;33m",  # Dimmed yellow
}

# Static list of valid categories representing various parts of the Llama Stack
# server codebase
CATEGORIES = [
    "core",
    "server",
    "router",
    "inference",
    "agents",
    "safety",
    "eval",
    "tools",
    "client",
]

_logger = logging.getLogger("llama_stack")
_logger.propagate = False

_default_level = logging.INFO

# Category-level mapping (can be modified by environment variables)
_category_levels: Dict[str, int] = {}


class TerminalStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        self.is_tty = hasattr(self.stream, "isatty") and self.stream.isatty()

    def format(self, record):
        record.is_tty = self.is_tty
        return super().format(record)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and fixed-width level names"""

    def format(self, record):
        levelname = record.levelname
        # Use only time with milliseconds, not date
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm format

        file_info = f"{record.filename}:{record.lineno}"

        # Get category from extra if available
        category = getattr(record, "category", None)
        msg = record.getMessage()

        if getattr(record, "is_tty", False):
            color = COLORS.get(levelname, COLORS["RESET"])
            if category:
                category_formatted = f"{COLORS['YELLOW_DIM']}{category}{COLORS['RESET']} "
                formatted_msg = (
                    f"{color}{levelname:<7}{COLORS['RESET']} {COLORS['DIM']}{timestamp}{COLORS['RESET']} "
                    f"{file_info:<20} {category_formatted}{msg}"
                )
            else:
                formatted_msg = (
                    f"{color}{levelname:<7}{COLORS['RESET']} {COLORS['DIM']}{timestamp}{COLORS['RESET']}] "
                    f"{file_info:<20} {msg}"
                )
        else:
            if category:
                formatted_msg = f"{levelname:<7} {timestamp} {file_info:<20} [{category}] {msg}"
            else:
                formatted_msg = f"{levelname:<7} {timestamp} {file_info:<20} {msg}"

        return formatted_msg


def init(default_level: int = logging.INFO) -> None:
    global _default_level, _category_levels, _logger

    _default_level = default_level

    _logger.setLevel(logging.DEBUG)
    _logger.handlers = []  # Clear existing handlers

    # Add our custom handler with the colored formatter
    handler = TerminalStreamHandler()
    formatter = ColoredFormatter()
    handler.setFormatter(formatter)
    _logger.addHandler(handler)

    for category in CATEGORIES:
        _category_levels[category] = default_level

    env_config = os.environ.get("LLAMA_STACK_LOGGING", "")
    if env_config:
        for pair in env_config.split(";"):
            if not pair.strip():
                continue

            try:
                category, level = pair.split("=", 1)
                category = category.strip().lower()
                level = level.strip().lower()

                level_value = {
                    "debug": logging.DEBUG,
                    "info": logging.INFO,
                    "warning": logging.WARNING,
                    "warn": logging.WARNING,
                    "error": logging.ERROR,
                    "critical": logging.CRITICAL,
                }.get(level)

                if level_value is None:
                    _logger.warning(f"Unknown log level '{level}' for category '{category}'")
                    continue

                if category == "all":
                    for cat in CATEGORIES:
                        _category_levels[cat] = level_value
                else:
                    if category in CATEGORIES:
                        _category_levels[category] = level_value
                    else:
                        _logger.warning(f"Unknown logging category: {category}")

            except ValueError:
                _logger.warning(f"Invalid logging configuration: {pair}")


def _should_log(level: int, category: str) -> bool:
    category_level = _category_levels.get(category.lower(), _default_level)
    return level >= category_level


def _log(level: int, level_name: str, category: str, msg: str, *args, **kwargs) -> None:
    if _should_log(level, category):
        kwargs.setdefault("extra", {})["category"] = category.lower()
        getattr(_logger, level_name)(msg, *args, stacklevel=3, **kwargs)


def debug(category: str, msg: str, *args, **kwargs) -> None:
    _log(logging.DEBUG, "debug", category, msg, *args, **kwargs)


def info(category: str, msg: str, *args, **kwargs) -> None:
    _log(logging.INFO, "info", category, msg, *args, **kwargs)


def warning(category: str, msg: str, *args, **kwargs) -> None:
    _log(logging.WARNING, "warning", category, msg, *args, **kwargs)


def warn(category: str, msg: str, *args, **kwargs) -> None:
    warning(category, msg, *args, **kwargs)


def error(category: str, msg: str, *args, **kwargs) -> None:
    _log(logging.ERROR, "error", category, msg, *args, **kwargs)


def critical(category: str, msg: str, *args, **kwargs) -> None:
    _log(logging.CRITICAL, "critical", category, msg, *args, **kwargs)


def exception(category: str, msg: str, *args, **kwargs) -> None:
    if _should_log(logging.ERROR, category):
        kwargs.setdefault("extra", {})["category"] = category.lower()
        _logger.exception(msg, *args, stacklevel=2, **kwargs)
