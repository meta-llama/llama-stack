# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
import os
from logging.config import dictConfig
from typing import Dict

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

# Predefined categories
CATEGORIES = ["core", "server", "router", "inference", "agents", "safety", "eval", "tools", "client"]

# Initialize category levels with default level
_category_levels: Dict[str, int] = {category: DEFAULT_LOG_LEVEL for category in CATEGORIES}


def parse_environment_config(env_config: str) -> Dict[str, int]:
    """
    Parse the LLAMA_STACK_LOGGING environment variable and return a dictionary of category log levels.

    Parameters:
        env_config (str): The value of the LLAMA_STACK_LOGGING environment variable.

    Returns:
        Dict[str, int]: A dictionary mapping categories to their log levels.
    """
    category_levels = {}
    for pair in env_config.split(";"):
        if not pair.strip():
            continue

        try:
            category, level = pair.split("=", 1)
            category = category.strip().lower()
            level = level.strip().upper()  # Convert to uppercase for logging._nameToLevel

            level_value = logging._nameToLevel.get(level)
            if level_value is None:
                logging.warning(
                    f"Unknown log level '{level}' for category '{category}'. Falling back to default 'INFO'."
                )
                continue

            if category == "all":
                # Apply the log level to all categories and the root logger
                for cat in CATEGORIES:
                    category_levels[cat] = level_value
                # Set the root logger's level to the specified level
                category_levels["root"] = level_value
            elif category in CATEGORIES:
                category_levels[category] = level_value
                logging.info(f"Setting '{category}' category to level '{level}'.")
            else:
                logging.warning(f"Unknown logging category: {category}. No changes made.")

        except ValueError:
            logging.warning(f"Invalid logging configuration: '{pair}'. Expected format: 'category=level'.")

    return category_levels


def setup_logging(category_levels: Dict[str, int]) -> None:
    """
    Configure logging based on the provided category log levels.

    Parameters:
        category_levels (Dict[str, int]): A dictionary mapping categories to their log levels.
    """
    log_format = "%(asctime)s %(name)s:%(lineno)d [%(category)s]: %(message)s"

    class CategoryFilter(logging.Filter):
        """Ensure category is always present in log records."""

        def filter(self, record):
            if not hasattr(record, "category"):
                record.category = "uncategorized"  # Default to 'uncategorized' if no category found
            return True

    # Determine the root logger's level (default to WARNING if not specified)
    root_level = category_levels.get("root", logging.WARNING)

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "rich": {
                "()": logging.Formatter,  # Standard formatter (RichHandler handles colors)
                "format": log_format,
            }
        },
        "handlers": {
            "console": {
                "class": "rich.logging.RichHandler",
                "formatter": "rich",
                "rich_tracebacks": True,
                "show_time": False,  # We handle timestamps ourselves in the log_format
                "show_path": False,
                "filters": ["category_filter"],  # Ensures category is included
            }
        },
        "filters": {
            "category_filter": {
                "()": CategoryFilter,
            }
        },
        "loggers": {
            category: {
                "handlers": ["console"],
                "level": category_levels.get(category, DEFAULT_LOG_LEVEL),
                "propagate": False,  # Disable propagation to root logger
            }
            for category in CATEGORIES
        },
        "root": {
            "handlers": ["console"],
            "level": root_level,  # Set root logger's level dynamically
        },
    }
    dictConfig(logging_config)


def get_logger(name: str, category: str = "uncategorized") -> logging.LoggerAdapter:
    """
    Returns a logger with the specified name and category.
    If no category is provided, defaults to 'uncategorized'.

    Parameters:
        name (str): The name of the logger (e.g., module or filename).
        category (str): The category of the logger (default 'uncategorized').

    Returns:
        logging.LoggerAdapter: Configured logger with category support.
    """
    # Use the name as the logger's name
    logger = logging.getLogger(name)
    # Apply the category's log level to the logger
    logger.setLevel(_category_levels.get(category, DEFAULT_LOG_LEVEL))
    # Attach the category as extra context
    return logging.LoggerAdapter(logger, {"category": category})


# Parse environment variable and configure logging
env_config = os.environ.get("LLAMA_STACK_LOGGING", "")
if env_config:
    print(f"Environment variable LLAMA_STACK_LOGGING found: {env_config}")
    _category_levels.update(parse_environment_config(env_config))

setup_logging(_category_levels)
