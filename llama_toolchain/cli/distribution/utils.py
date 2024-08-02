# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import errno
import os
import pty
import select
import subprocess
import sys
import termios
import tty


def run_with_pty(command):
    old_settings = termios.tcgetattr(sys.stdin)

    # Create a new pseudo-terminal
    master, slave = pty.openpty()

    try:
        # ensure the terminal does not echo input
        tty.setraw(sys.stdin.fileno())

        process = subprocess.Popen(
            command,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            universal_newlines=True,
        )

        # Close the slave file descriptor as it's now owned by the subprocess
        os.close(slave)

        def handle_io():
            while True:
                rlist, _, _ = select.select([sys.stdin, master], [], [])

                if sys.stdin in rlist:
                    data = os.read(sys.stdin.fileno(), 1024)
                    if not data:  # EOF
                        break
                    os.write(master, data)

                if master in rlist:
                    data = os.read(master, 1024)
                    if not data:
                        break
                    os.write(sys.stdout.fileno(), data)

        handle_io()
    except (EOFError, KeyboardInterrupt):
        pass
    except OSError as e:
        if e.errno != errno.EIO:
            raise
    finally:
        # Restore original terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    process.wait()
    os.close(master)

    return process.returncode


def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    if process.returncode != 0:
        print(f"Error: {error.decode('utf-8')}")
        sys.exit(1)
    return output.decode("utf-8")
