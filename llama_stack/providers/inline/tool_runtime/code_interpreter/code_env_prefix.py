# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import errno

# Disabling potentially dangerous functions
import os as _os
from functools import partial

os_funcs_to_disable = [
    "kill",
    "system",
    "putenv",
    "remove",
    "removedirs",
    "rmdir",
    "fchdir",
    "setuid",
    "fork",
    "forkpty",
    "killpg",
    "rename",
    "renames",
    "truncate",
    "replace",
    # "unlink",  # Commenting as this was blocking matpltlib from rendering plots correctly
    "fchmod",
    "fchown",
    "chmod",
    "chown",
    "chroot",
    "fchdir",
    "lchflags",
    "lchmod",
    "lchown",
    "chdir",
]


def call_not_allowed(*args, **kwargs):
    raise OSError(errno.EPERM, "Call are not permitted in this environment")


for func_name in os_funcs_to_disable:
    if hasattr(_os, func_name):
        setattr(_os, func_name, partial(call_not_allowed, _func_name=f"os.{func_name}"))

import shutil as _shutil

for func_name in ["rmtree", "move", "chown"]:
    if hasattr(_shutil, func_name):
        setattr(
            _shutil,
            func_name,
            partial(call_not_allowed, _func_name=f"shutil.{func_name}"),
        )

import subprocess as _subprocess


def popen_not_allowed(*args, **kwargs):
    raise _subprocess.CalledProcessError(
        -1,
        args[0] if args else "unknown",
        stderr="subprocess.Popen is not allowed in this environment",
    )


_subprocess.Popen = popen_not_allowed


import atexit as _atexit
import builtins as _builtins
import io as _io
import json as _json
import sys as _sys

# NB! The following "unused" imports crucial, make sure not not to remove
# them with linters - they're used in code_execution.py
from contextlib import (  # noqa
    contextmanager as _contextmanager,
    redirect_stderr as _redirect_stderr,
    redirect_stdout as _redirect_stdout,
)
from multiprocessing.connection import Connection as _Connection

# Mangle imports to avoid polluting model execution namespace.

_IO_SINK = _io.StringIO()
_NETWORK_TIMEOUT = 5
_NETWORK_CONNECTIONS = None


def _open_connections():
    global _NETWORK_CONNECTIONS
    if _NETWORK_CONNECTIONS is not None:
        # Ensure connections only opened once.
        return _NETWORK_CONNECTIONS
    req_w_fd, resp_r_fd = _sys.argv[1], _sys.argv[2]
    req_con = _Connection(int(req_w_fd), readable=False)
    resp_con = _Connection(int(resp_r_fd), writable=False)
    _NETWORK_CONNECTIONS = (req_con, resp_con)
    return _NETWORK_CONNECTIONS


_builtins._open_connections = _open_connections


@_atexit.register
def _close_connections():
    global _NETWORK_CONNECTIONS
    if _NETWORK_CONNECTIONS is None:
        return
    for con in _NETWORK_CONNECTIONS:
        con.close()
    del _NETWORK_CONNECTIONS


def _network_call(request):
    # NOTE: We communicate with the parent process in json, encoded
    # in raw bytes. We do this because native send/recv methods use
    # pickle which involves execution of arbitrary code.
    _open_connections()
    req_con, resp_con = _NETWORK_CONNECTIONS

    req_con.send_bytes(_json.dumps(request).encode("utf-8"))
    if resp_con.poll(timeout=_NETWORK_TIMEOUT) is None:
        raise Exception(f"Network request timed out: {_json.dumps(request)}")
    else:
        return _json.loads(resp_con.recv_bytes().decode("utf-8"))
