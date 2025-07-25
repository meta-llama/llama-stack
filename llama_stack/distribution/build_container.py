#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request

import yaml


class Colors:
    RED = "\033[0;31m"
    NC = "\033[0m"  # No Color


def is_command_available(command: str) -> bool:
    """Check if a command is available in the system."""
    return shutil.which(command) is not None


class ContainerBuilder:
    def __init__(self):
        # Environment variables with defaults
        self.llama_stack_dir = os.getenv("LLAMA_STACK_DIR", "")
        self.llama_stack_client_dir = os.getenv("LLAMA_STACK_CLIENT_DIR", "")
        self.test_pypi_version = os.getenv("TEST_PYPI_VERSION", "")
        self.pypi_version = os.getenv("PYPI_VERSION", "")
        self.build_platform = os.getenv("BUILD_PLATFORM", "")
        self.uv_http_timeout = os.getenv("UV_HTTP_TIMEOUT", "500")
        self.use_copy_not_mount = os.getenv("USE_COPY_NOT_MOUNT", "")
        self.mount_cache = os.getenv("MOUNT_CACHE", "--mount=type=cache,id=llama-stack-cache,target=/root/.cache")
        self.container_binary = os.getenv("CONTAINER_BINARY", "docker")
        self.container_opts = os.getenv("CONTAINER_OPTS", "--progress=plain")

        # Constants
        self.run_config_path = "/app/run.yaml"
        self.build_context_dir = os.getcwd()
        self.stack_mount = "/app/llama-stack-source"
        self.client_mount = "/app/llama-stack-client-source"

        # Temporary directory and Containerfile
        self.temp_dir = tempfile.mkdtemp()
        self.containerfile_path = os.path.join(self.temp_dir, "Containerfile")

    def cleanup(self):
        """Clean up temporary files."""

        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temporary directory: {e}", file=sys.stderr)

        # Clean up copied files in build context
        run_yaml_path = os.path.join(self.build_context_dir, "run.yaml")
        if os.path.exists(run_yaml_path):
            try:
                os.remove(run_yaml_path)
            except Exception as e:
                print(f"Warning: Could not clean up run.yaml: {e}", file=sys.stderr)

    def add_to_container(self, content: str):
        """Add content to the Containerfile."""
        with open(self.containerfile_path, "a") as f:
            f.write(content + "\n")

    def validate_args(self, args):
        """Validate command line arguments."""
        if not is_command_available(self.container_binary):
            print(
                f"{Colors.RED}Error: {self.container_binary} command not found. "
                f"Is {self.container_binary} installed and in your PATH?{Colors.NC}",
                file=sys.stderr,
            )
            sys.exit(1)

    def generate_base_image_setup(self, container_base: str):
        """Generate the base image setup commands."""
        self.add_to_container(f"""FROM {container_base}
WORKDIR /app""")

        if "registry.access.redhat.com/ubi9" in container_base:
            self.add_to_container("""# We install the Python 3.12 dev headers and build tools so that any
# C-extension wheels (e.g. polyleven, faiss-cpu) can compile successfully.

RUN dnf -y update && dnf install -y iputils git net-tools wget \\
    vim-minimal python3.12 python3.12-pip python3.12-wheel \\
    python3.12-setuptools python3.12-devel gcc make && \\
    ln -s /bin/pip3.12 /bin/pip && ln -s /bin/python3.12 /bin/python && dnf clean all""")
        else:
            self.add_to_container("""RUN apt-get update && apt-get install -y \\
       iputils-ping net-tools iproute2 dnsutils telnet \\
       curl wget telnet git\\
       procps psmisc lsof \\
       traceroute \\
       bubblewrap \\
       gcc \\
       && rm -rf /var/lib/apt/lists/*""")

        self.add_to_container("""ENV UV_SYSTEM_PYTHON=1
RUN pip install uv""")

    def add_pip_dependencies(self, pip_dependencies: str, special_pip_deps: str):
        """Add pip dependencies to the container."""
        # Set link mode to copy
        self.add_to_container("ENV UV_LINK_MODE=copy")

        # Add regular pip dependencies
        if pip_dependencies:
            pip_args = shlex.split(pip_dependencies)
            quoted_deps = " ".join(shlex.quote(dep) for dep in pip_args)
            self.add_to_container(f"RUN {self.mount_cache} uv pip install {quoted_deps}")

        # Add special pip dependencies
        if special_pip_deps:
            parts = special_pip_deps.split("#")
            for part in parts:
                if part.strip():
                    pip_args = shlex.split(part.strip())
                    quoted_deps = " ".join(shlex.quote(dep) for dep in pip_args)
                    self.add_to_container(f"RUN {self.mount_cache} uv pip install {quoted_deps}")

    def handle_run_config(self, run_config: str):
        """Handle run configuration file."""
        if not run_config:
            return

        # Copy the run config to the build context
        run_yaml_dest = os.path.join(self.build_context_dir, "run.yaml")
        shutil.copy2(run_config, run_yaml_dest)

        # Parse the run.yaml configuration for external providers
        try:
            with open(run_config) as f:
                config = yaml.safe_load(f)

            external_providers_dir = config.get("external_providers_dir", "")
            if external_providers_dir:
                # Expand environment variables in path
                external_providers_dir = os.path.expandvars(external_providers_dir)

                if os.path.isdir(external_providers_dir):
                    print(f"Copying external providers directory: {external_providers_dir}")
                    providers_dest = os.path.join(self.build_context_dir, "providers.d")
                    shutil.copytree(external_providers_dir, providers_dest, dirs_exist_ok=True)

                    self.add_to_container("COPY providers.d /.llama/providers.d")

                    # Update the run.yaml file to change external_providers_dir
                    with open(run_yaml_dest) as f:
                        content = f.read()

                    # Replace external_providers_dir line
                    content = re.sub(
                        r"external_providers_dir:.*", "external_providers_dir: /.llama/providers.d", content
                    )

                    with open(run_yaml_dest, "w") as f:
                        f.write(content)

        except Exception as e:
            print(f"Warning: Could not parse run.yaml: {e}", file=sys.stderr)

        # Copy run config into docker image
        self.add_to_container(f"COPY run.yaml {self.run_config_path}")

    def install_local_package(self, directory: str, mount_point: str, name: str):
        """Install a local package in the container."""
        if not os.path.isdir(directory):
            print(
                f"{Colors.RED}Warning: {name} is set but directory does not exist: {directory}{Colors.NC}",
                file=sys.stderr,
            )
            sys.exit(1)

        if self.use_copy_not_mount == "true":
            self.add_to_container(f"COPY {directory} {mount_point}")

        self.add_to_container(f"RUN {self.mount_cache} uv pip install -e {mount_point}")

    def install_llama_stack(self):
        """Install llama-stack package."""
        if self.llama_stack_client_dir:
            self.install_local_package(self.llama_stack_client_dir, self.client_mount, "LLAMA_STACK_CLIENT_DIR")

        if self.llama_stack_dir:
            self.install_local_package(self.llama_stack_dir, self.stack_mount, "LLAMA_STACK_DIR")
        else:
            if self.test_pypi_version:
                # Install damaged packages first for test-pypi
                self.add_to_container(f"RUN {self.mount_cache} uv pip install fastapi libcst")
                self.add_to_container(f"""RUN {self.mount_cache} uv pip install --extra-index-url https://test.pypi.org/simple/ \\
  --index-strategy unsafe-best-match \\
  llama-stack=={self.test_pypi_version}""")
            else:
                if self.pypi_version:
                    spec_version = f"llama-stack=={self.pypi_version}"
                else:
                    spec_version = "llama-stack"

                self.add_to_container(f"RUN {self.mount_cache} uv pip install {spec_version}")

    def add_entrypoint(self, template_or_config: str, run_config: str):
        """Add the container entrypoint."""
        # Remove uv after installation
        self.add_to_container("RUN pip uninstall -y uv")

        # Set entrypoint based on configuration
        if run_config:
            self.add_to_container(
                f'ENTRYPOINT ["python", "-m", "llama_stack.distribution.server.server", "--config", "{self.run_config_path}"]'
            )
        elif not template_or_config.endswith(".yaml"):
            self.add_to_container(
                f'ENTRYPOINT ["python", "-m", "llama_stack.distribution.server.server", "--template", "{template_or_config}"]'
            )

        # Add generic container setup
        self.add_to_container("""RUN mkdir -p /.llama /.cache && chmod -R g+rw /app /.llama /.cache""")

    def get_version_tag(self):
        """Get the version tag for the image."""
        if self.pypi_version:
            return self.pypi_version
        elif self.test_pypi_version:
            return f"test-{self.test_pypi_version}"
        elif self.llama_stack_dir or self.llama_stack_client_dir:
            return "dev"
        else:
            try:
                url = "https://pypi.org/pypi/llama-stack/json"
                with urllib.request.urlopen(url) as response:
                    data = json.loads(response.read())
                    return data["info"]["version"]
            except Exception:
                return "latest"

    def build_container(self, image_name: str) -> tuple[list[str], str]:
        """Build the container and return CLI arguments."""
        cli_args = shlex.split(self.container_opts)

        # Add volume mounts if not using copy mode
        if self.use_copy_not_mount != "true":
            if self.llama_stack_dir:
                abs_path = os.path.abspath(self.llama_stack_dir)
                cli_args.extend(["-v", f"{abs_path}:{self.stack_mount}"])
            if self.llama_stack_client_dir:
                abs_path = os.path.abspath(self.llama_stack_client_dir)
                cli_args.extend(["-v", f"{abs_path}:{self.client_mount}"])

        # Handle SELinux if available
        try:
            if is_command_available("selinuxenabled"):
                result = subprocess.run(["selinuxenabled"], capture_output=True)
                if result.returncode == 0:
                    cli_args.extend(["--security-opt", "label=disable"])
        except Exception:
            pass

        # Set platform
        arch = platform.machine()
        if self.build_platform:
            cli_args.extend(["--platform", self.build_platform])
        elif arch in ["arm64", "aarch64"]:
            cli_args.extend(["--platform", "linux/arm64"])
        elif arch == "x86_64":
            cli_args.extend(["--platform", "linux/amd64"])
        else:
            print(f"Unsupported architecture: {arch}")
            sys.exit(1)

        # Create image tag
        version_tag = self.get_version_tag()
        image_tag = f"{image_name}:{version_tag}"

        return cli_args, image_tag

    def run_build(self, cli_args: list[str], image_tag: str):
        """Execute the container build command."""
        print(f"PWD: {os.getcwd()}")
        print(f"Containerfile: {self.containerfile_path}")

        # Print Containerfile content
        print(f"Containerfile created successfully in {self.containerfile_path}\n")
        with open(self.containerfile_path) as f:
            print(f.read())
        print()

        # Build the container
        cmd = [
            self.container_binary,
            "build",
            *cli_args,
            "-t",
            image_tag,
            "-f",
            self.containerfile_path,
            self.build_context_dir,
        ]

        print("Running command:")
        print(" ".join(shlex.quote(arg) for arg in cmd))

        try:
            subprocess.run(cmd, check=True)
            print("Success!")
        except subprocess.CalledProcessError as e:
            print(f"Build failed with exit code {e.returncode}")
            sys.exit(e.returncode)


def main():
    if len(sys.argv) < 5:
        print(
            "Usage: build_container.py <template_or_config> <image_name> <container_base> <pip_dependencies> [<run_config>] [<special_pip_deps>]",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Build container images for llama-stack", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("template_or_config", help="Template name or path to config file")
    parser.add_argument("image_name", help="Name for the container image")
    parser.add_argument("container_base", help="Base container image")
    parser.add_argument("pip_dependencies", help="Pip dependencies to install")
    parser.add_argument("run_config", nargs="?", default="", help="Optional path to run.yaml config file")
    parser.add_argument("special_pip_deps", nargs="?", default="", help="Optional special pip dependencies")

    args = parser.parse_args()

    # Handle the complex argument parsing logic from the bash script
    # If we have 5+ args and the 5th arg doesn't end with .yaml, it's special_pip_deps
    if len(sys.argv) >= 6:
        if not sys.argv[5].endswith(".yaml"):
            args.special_pip_deps = args.run_config
            args.run_config = ""

    builder = ContainerBuilder()
    try:
        builder.validate_args(args)

        # Generate Containerfile
        builder.generate_base_image_setup(args.container_base)
        builder.add_pip_dependencies(args.pip_dependencies, args.special_pip_deps)
        builder.handle_run_config(args.run_config)
        builder.install_llama_stack()
        builder.add_entrypoint(args.template_or_config, args.run_config)

        # Build container
        cli_args, image_tag = builder.build_container(args.image_name)
        builder.run_build(cli_args, image_tag)

    finally:
        builder.cleanup()


if __name__ == "__main__":
    main()
