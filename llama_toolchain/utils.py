# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import getpass
import os
from typing import Optional

from hydra import compose, initialize, MissingConfigException
from hydra.core.global_hydra import GlobalHydra

from omegaconf import OmegaConf


LLAMA_STACK_CONFIG_DIR = os.path.expanduser("~/.llama/")


def get_root_directory():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while os.path.isfile(os.path.join(current_dir, "__init__.py")):
        current_dir = os.path.dirname(current_dir)

    return current_dir


def get_default_config_dir():
    return os.path.join(LLAMA_STACK_CONFIG_DIR, "configs")


def parse_config(config_dir: str, config_path: Optional[str] = None) -> str:
    # Configs can be
    # 1. relative paths in {config_dir}/
    # 2. or default to file {config_dir}/{user}.yaml
    # 3. or ultimate default to {config_dir}/default.yaml

    # Get the relative path from the current file to the config directory
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.relpath(config_dir, current_file_directory)

    GlobalHydra.instance().clear()
    initialize(config_path=relative_path)

    if config_path is None:
        try:
            user = getpass.getuser()
            config_name = user
        except MissingConfigException:
            print(f"No user-specific {user}.yaml, using default")
            config_name = "default"
    else:
        config_name = config_path

    config_abs_path = os.path.abspath(os.path.join(config_dir, f"{config_name}.yaml"))
    print(f"Loading config from : {config_abs_path}")
    config = compose(config_name=config_name)

    print("Yaml config:")
    print("------------------------")
    print(OmegaConf.to_yaml(config, resolve=True))
    print("------------------------")

    return config
