# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
import os
import signal
import subprocess

from termcolor import cprint

log = logging.getLogger(__name__)

import importlib
import json
from pathlib import Path

from llama_stack.distribution.utils.image_types import LlamaStackImageType


def formulate_run_args(image_type, image_name, config, template_name) -> list:
    env_name = ""
    if image_type == LlamaStackImageType.CONTAINER.value or config.container_image:
        env_name = f"distribution-{template_name}" if template_name else config.container_image
    elif image_type == LlamaStackImageType.CONDA.value:
        current_conda_env = os.environ.get("CONDA_DEFAULT_ENV")
        env_name = image_name or current_conda_env
        if not env_name:
            cprint(
                "No current conda environment detected, please specify a conda environment name with --image-name",
                color="red",
            )
            return

        def get_conda_prefix(env_name):
            # Conda "base" environment does not end with "base" in the
            # prefix, so should be handled separately.
            if env_name == "base":
                return os.environ.get("CONDA_PREFIX")
            # Get conda environments info
            conda_env_info = json.loads(subprocess.check_output(["conda", "info", "--envs", "--json"]).decode())
            envs = conda_env_info["envs"]
            for envpath in envs:
                if os.path.basename(envpath) == env_name:
                    return envpath
            return None

        print(f"Using conda environment: {env_name}")
        conda_prefix = get_conda_prefix(env_name)
        if not conda_prefix:
            cprint(
                f"Conda environment {env_name} does not exist.",
                color="red",
            )
            return

        build_file = Path(conda_prefix) / "llamastack-build.yaml"
        if not build_file.exists():
            cprint(
                f"Build file {build_file} does not exist.\n\nPlease run `llama stack build` or specify the correct conda environment name with --image-name",
                color="red",
            )
            return
    else:
        # else must be venv since that is the only valid option left.
        current_venv = os.environ.get("VIRTUAL_ENV")
        env_name = image_name or current_venv
        if not env_name:
            cprint(
                "No current virtual environment detected, please specify a virtual environment name with --image-name",
                color="red",
            )
            return
        print(f"Using virtual environment: {env_name}")

    script = importlib.resources.files("llama_stack") / "distribution/start_stack.sh"
    run_args = [
        script,
        image_type,
        env_name,
    ]

    return run_args


def in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def run_command(command: list[str]) -> int:
    """
    Run a command with interrupt handling and output capture.
    Uses subprocess.run with direct stream piping for better performance.

    Args:
        command (list): The command to run.

    Returns:
        int: The return code of the command.
    """
    original_sigint = signal.getsignal(signal.SIGINT)
    ctrl_c_pressed = False

    def sigint_handler(signum, frame):
        nonlocal ctrl_c_pressed
        ctrl_c_pressed = True
        log.info("\nCtrl-C detected. Aborting...")

    try:
        # Set up the signal handler
        signal.signal(signal.SIGINT, sigint_handler)

        # Run the command with stdout/stderr piped directly to system streams
        result = subprocess.run(
            command,
            text=True,
            check=False,
        )
        return result.returncode
    except subprocess.SubprocessError as e:
        log.error(f"Subprocess error: {e}")
        return 1
    except Exception as e:
        log.exception(f"Unexpected error: {e}")
        return 1
    finally:
        # Restore the original signal handler
        signal.signal(signal.SIGINT, original_sigint)
