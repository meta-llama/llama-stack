# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import importlib
import platform
import subprocess
import sys

import psutil

from llama_stack.cli.subcommand import Subcommand


def _system_info():
    sys_info = {
        "sys.version": sys.version,
        "sys.platform": sys.platform,
        "platform.machine": platform.machine(),
        "platform.node": platform.node(),
        "platform.python_version": platform.python_version(),
    }
    if sys.platform == "linux":
        os_release = platform.freedesktop_os_release()
        for key in ["ID", "VERSION_ID", "PRETTY_NAME"]:
            value = os_release.get(key)
            if value:
                sys_info[f"os-release.{key}"] = value
    elif sys.platform == "darwin":
        try:
            cpu_info = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode("utf-8").strip()
            sys_info["platform.cpu_brand"] = cpu_info
        except subprocess.CalledProcessError:
            sys_info["platform.cpu_brand"] = "Unknown"
    sys_memory_info = psutil.virtual_memory()
    sys_info["memory.used"] = f"{(sys_memory_info.used / 1024**3):.2f} GB"
    sys_info["memory.available"] = f"{(sys_memory_info.available / 1024**3):.2f} GB"
    sys_info["memory.total"] = f"{(sys_memory_info.total / 1024**3):.2f} GB"
    return sys_info


def _list_pkg_version(pkg_name):
    packages = sorted(
        (distribution.name, distribution.version)
        for distribution in importlib.metadata.distributions()
        if distribution.name.startswith(pkg_name)
    )
    return {f"{name}.version": version for name, version in packages}


def _cuda_info():
    import torch

    if torch.cuda.is_available():
        cuda_info = {
            "cuda.version": torch.version.cuda,
            "gpu.count": torch.cuda.device_count(),
            "cuda.bf16": torch.cuda.is_bf16_supported(),
            "cuda.current.device": torch.cuda.current_device(),
        }
        for idx in range(torch.cuda.device_count()):
            device = torch.device("cuda", idx)
            free, total = torch.cuda.mem_get_info(device)
            cuda_info[f"cuda.{idx}.name"] = torch.cuda.get_device_name(device)
            cuda_info[f"cuda.{idx}.free"] = f"{(free / 1024**3):.1f} GB"
            cuda_info[f"cuda.{idx}.total"] = f"{(total / 1024**3):.1f} GB"
        return cuda_info


def _add_to_group(groups, item_name, info):
    if item_name not in groups:
        groups[item_name] = []
    groups[item_name].extend(info.items())


def _get_env_info_by_group():
    groups = {}
    _add_to_group(groups, "system", _system_info())
    _add_to_group(groups, "llama packages", _list_pkg_version("llama"))
    _add_to_group(groups, "torch packages", _list_pkg_version("torch"))

    # only cuda exist
    cuda_info = _cuda_info()
    if cuda_info:
        _add_to_group(groups, "cuda info", _cuda_info())

    return groups


class LlamaStackInfo(Subcommand):
    """Llama cli for showing llama stack environment information"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "info",
            prog="llama info",
            description="Show llama stack environment information",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_llama_stack_info_cmd)

    def _run_llama_stack_info_cmd(self, args: argparse.Namespace) -> None:
        groups = _get_env_info_by_group()
        for idx, (group, items) in enumerate(groups.items()):
            if idx > 0:
                print()
            print(f"{group}:")
            for k, v in items:
                print(f"  {k}: {v}")
