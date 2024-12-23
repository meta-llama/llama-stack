# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import json
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List

from PIL import Image

from .utils import get_code_env_prefix

TOOLS_ATTACHMENT_KEY = "__tools_attachment__"
TOOLS_ATTACHMENT_KEY_REGEX = re.compile(r"__tools_attachment__=(\{.*?\})")

DIRNAME = Path(__file__).parent

CODE_EXEC_TIMEOUT = 20
CODE_ENV_PREFIX = get_code_env_prefix()

STDOUTERR_SINK_WRAPPER_TEMPLATE = """\
with _redirect_stdout(_IO_SINK), _redirect_stderr(_IO_SINK):
{code}\
"""

TRYEXCEPT_WRAPPER_TEMPLATE = """\
try:
{code}
except:
    pass\
"""


def generate_bwrap_command(bind_dirs: List[str]) -> str:
    """
    Generate the bwrap command string for binding all
    directories in the current directory read-only.
    """
    bwrap_args = ""
    bwrap_args += "--ro-bind / / "
    # Add the --dev flag to mount device files
    bwrap_args += "--dev /dev "
    for d in bind_dirs:
        bwrap_args += f"--bind {d} {d} "

    # Add the --unshare-all flag to isolate the sandbox from the rest of the system
    bwrap_args += "--unshare-all "
    # Add the --die-with-parent flag to ensure the child process dies when bwrap's parent dies
    bwrap_args += "--die-with-parent "
    return bwrap_args


@dataclass
class CodeExecutionContext:
    matplotlib_dump_dir: str
    use_proxy: bool = False


@dataclass
class CodeExecutionRequest:
    scripts: List[str]
    only_last_cell_stdouterr: bool = True
    only_last_cell_fail: bool = True
    seed: int = 0
    strip_fpaths_in_stderr: bool = True


class CodeExecutor:
    def __init__(self, context: CodeExecutionContext):
        self.context = context

    def execute(self, req: CodeExecutionRequest) -> dict:
        scripts = req.scripts
        for i in range(len(scripts) - 1):
            if req.only_last_cell_stdouterr:
                scripts[i] = STDOUTERR_SINK_WRAPPER_TEMPLATE.format(
                    code=textwrap.indent(scripts[i], " " * 4)
                )
            if req.only_last_cell_fail:
                scripts[i] = TRYEXCEPT_WRAPPER_TEMPLATE.format(
                    code=textwrap.indent(scripts[i], " " * 4)
                )

        # Seeds prefix:
        seed = req.seed
        seeds_prefix = f"""\
def _set_seeds():
    import random
    random.seed({seed})
    import numpy as np
    np.random.seed({seed})
_set_seeds()\
"""

        script = "\n\n".join([seeds_prefix] + [CODE_ENV_PREFIX] + scripts)
        with tempfile.TemporaryDirectory() as dpath:
            bwrap_prefix = "bwrap " + generate_bwrap_command(bind_dirs=[dpath])
            cmd = [*bwrap_prefix.split(), sys.executable, "-c", script]
            code_fpath = os.path.join(dpath, "code.py")
            with open(code_fpath, "w") as f:
                f.write(script)

            try:
                python_path = os.environ.get("PYTHONPATH", "")
                env = dict(
                    os.environ,
                    PYTHONHASHSEED=str(seed),
                    MPLCONFIGDIR=dpath,
                    MPLBACKEND="module://matplotlib_custom_backend",
                    PYTHONPATH=f"{DIRNAME}:{python_path}",
                )
                stdout, stderr, returncode = do_subprocess(
                    cmd=cmd,
                    env=env,
                    ctx=self.context,
                )

                stderr = stderr.strip()
                if req.strip_fpaths_in_stderr:
                    pattern = r'File "([^"]+)", line (\d+)'
                    stderr = re.sub(pattern, r"line \2", stderr)

                return {
                    "process_status": "completed",
                    "returncode": returncode,
                    "stdout": stdout.strip(),
                    "stderr": stderr,
                }

            except subprocess.TimeoutExpired:
                return {
                    "process_status": "timeout",
                    "stdout": "Timed out",
                    "stderr": "Timed out",
                }

            except Exception as e:
                return {
                    "process_status": "error",
                    "error_type": type(e).__name__,
                    "stderr": str(e),
                    "stdout": str(e),
                }


def process_matplotlib_response(response, matplotlib_dump_dir: str):
    image_data = response["image_data"]
    # Convert the base64 string to a bytes object
    images = [base64.b64decode(d["image_base64"]) for d in image_data]
    # Create a list of PIL images from the bytes objects
    images = [Image.open(BytesIO(img)) for img in images]
    # Create a list of image paths
    image_paths = []
    for i, img in enumerate(images):
        # create new directory for each day to better organize data:
        dump_dname = datetime.today().strftime("%Y-%m-%d")
        dump_dpath = Path(matplotlib_dump_dir, dump_dname)
        dump_dpath.mkdir(parents=True, exist_ok=True)
        # save image into a file
        dump_fname = f"matplotlib_{str(time.time()).replace('.', '_')}_{i}.png"
        dump_fpath = dump_dpath / dump_fname
        img.save(dump_fpath, "PNG")
        image_paths.append(str(dump_fpath))

    # this is kind of convoluted, we send back this response to the subprocess which
    # prints it out
    info = {
        "filepath": str(image_paths[-1]),
        "mimetype": "image/png",
    }
    return f"{TOOLS_ATTACHMENT_KEY}={json.dumps(info)}"


def execute_subprocess_request(request, ctx: CodeExecutionContext):
    "Route requests from the subprocess (via network Pipes) to the internet/tools."
    if request["type"] == "matplotlib":
        return process_matplotlib_response(request, ctx.matplotlib_dump_dir)
    else:
        raise Exception(f'Unrecognised network request type: {request["type"]}')


def do_subprocess(*, cmd: list, env: dict, ctx: CodeExecutionContext):
    # Create Pipes to be used for any external tool/network requests.
    req_r, req_w = multiprocessing.Pipe(duplex=False)
    resp_r, resp_w = multiprocessing.Pipe(duplex=False)

    cmd += [str(req_w.fileno()), str(resp_r.fileno())]
    proc = subprocess.Popen(
        cmd,
        pass_fds=(req_w.fileno(), resp_r.fileno()),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        env=env,
    )

    # Close unnecessary fds.
    req_w.close()
    resp_r.close()

    pipe_close = False
    done_read = False
    start = time.monotonic()
    while proc.poll() is None and not pipe_close:
        if req_r.poll(0.1):
            # NB: Python pipe semantics for poll and recv mean that
            # poll() returns True is a pipe is closed.
            # CF old school PEP from '09
            #  https://bugs.python.org/issue5573
            try:
                request = json.loads(req_r.recv_bytes().decode("utf-8"))
                response = execute_subprocess_request(request, ctx)

                resp_w.send_bytes(json.dumps(response).encode("utf-8"))
            except EOFError:
                # The request pipe is closed - set a marker to exit
                # after the next attempt at reading stdout/stderr.
                pipe_close = True

            try:
                # If lots has been printed, pipe might be full but
                # proc cannot exit until all the stdout/stderr
                # been written/read.
                stdout, stderr = proc.communicate(timeout=0.3)
                done_read = True
            except subprocess.TimeoutExpired:
                # The program has not terminated. Ignore it, there
                # may be more network/tool requests.
                continue
        if time.monotonic() - start > CODE_EXEC_TIMEOUT:
            proc.terminate()
            raise subprocess.TimeoutExpired(cmd, CODE_EXEC_TIMEOUT)

    if not done_read:
        # Solve race condition where process terminates before
        # we hit the while loop.
        stdout, stderr = proc.communicate(timeout=0.3)

    resp_w.close()
    req_r.close()
    return stdout, stderr, proc.returncode
