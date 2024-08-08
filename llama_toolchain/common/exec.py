# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import errno
import os
import pty
import select
import signal
import subprocess
import sys
import termios

from termcolor import cprint


# run a command in a pseudo-terminal, with interrupt handling,
# useful when you want to run interactive things
def run_with_pty(command):
    master, slave = pty.openpty()

    old_settings = termios.tcgetattr(sys.stdin)
    original_sigint = signal.getsignal(signal.SIGINT)

    ctrl_c_pressed = False

    def sigint_handler(signum, frame):
        nonlocal ctrl_c_pressed
        ctrl_c_pressed = True
        cprint("\nCtrl-C detected. Aborting...", "white", attrs=["bold"])

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
        if process.poll() is None:
            process.terminate()
            process.wait()

    return process.returncode


def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    if process.returncode != 0:
        print(f"Error: {error.decode('utf-8')}")
        sys.exit(1)
    return output.decode("utf-8")
