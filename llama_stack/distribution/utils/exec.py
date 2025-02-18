# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import errno
import logging
import os
import select
import signal
import subprocess
import sys

log = logging.getLogger(__name__)


def run_with_pty(command):
    if sys.platform.startswith("win"):
        return _run_with_pty_win(command)
    else:
        return _run_with_pty_unix(command)


# run a command in a pseudo-terminal, with interrupt handling,
# useful when you want to run interactive things
def _run_with_pty_unix(command):
    import pty
    import termios

    master, slave = pty.openpty()

    old_settings = termios.tcgetattr(sys.stdin)
    original_sigint = signal.getsignal(signal.SIGINT)

    ctrl_c_pressed = False
    process = None

    def sigint_handler(signum, frame):
        nonlocal ctrl_c_pressed
        ctrl_c_pressed = True
        log.info("\nCtrl-C detected. Aborting...")

    try:
        # Set up the signal handler
        signal.signal(signal.SIGINT, sigint_handler)

        new_settings = termios.tcgetattr(sys.stdin)
        new_settings[3] = new_settings[3] & ~termios.ECHO  # Disable echo
        new_settings[3] = new_settings[3] & ~termios.ICANON  # Disable canonical mode
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)

        process = subprocess.Popen(
            command,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            universal_newlines=True,
            preexec_fn=os.setsid,
        )

        # Close the slave file descriptor as it's now owned by the subprocess
        os.close(slave)

        def handle_io():
            while not ctrl_c_pressed:
                try:
                    rlist, _, _ = select.select([sys.stdin, master], [], [], 0.1)

                    if sys.stdin in rlist:
                        data = os.read(sys.stdin.fileno(), 1024)
                        if not data:
                            break
                        os.write(master, data)

                    if master in rlist:
                        data = os.read(master, 1024)
                        if not data:
                            break
                        sys.stdout.buffer.write(data)
                        sys.stdout.flush()

                except KeyboardInterrupt:
                    # This will be raised when Ctrl+C is pressed
                    break

                if process.poll() is not None:
                    break

        handle_io()
    except (EOFError, KeyboardInterrupt):
        pass
    except OSError as e:
        if e.errno != errno.EIO:
            raise
    finally:
        # Clean up
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        signal.signal(signal.SIGINT, original_sigint)

        os.close(master)
        if process and process.poll() is None:
            process.terminate()
            process.wait()

    return process.returncode


# run a command in a pseudo-terminal in windows, with interrupt handling,
def _run_with_pty_win(command):
    """
    Runs a command with interactive support using subprocess directly.
    """
    try:
        # For shell scripts on Windows, use appropriate shell
        if isinstance(command, (list, tuple)):
            if command[0].endswith(".sh"):
                if os.path.exists("/usr/bin/bash"):  # WSL
                    command = ["bash"] + command
                else:
                    # Use cmd.exe with bash while preserving all arguments
                    command = ["cmd.exe", "/c", "bash"] + command

        process = subprocess.Popen(
            command,
            shell=True,
            universal_newlines=True,
        )

        process.wait()

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    finally:
        if process and process.poll() is None:
            process.terminate()
            process.wait()
    return process.returncode


def run_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Script Output\n", result.stdout)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print("Error running script:", e)
        print("Error output:", e.stderr)
        return e.returncode
