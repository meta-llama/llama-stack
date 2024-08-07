# llama-toolchain

[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-toolchain)](https://pypi.org/project/llama-toolchain/)
[![Discord](https://img.shields.io/discord/1257833999603335178)](https://discord.gg/TZAAYNVtrU)

This repo contains the API specifications for various components of the Llama Stack as well implementations for some of those APIs like model inference.

The Llama Stack consists of toolchain-apis and agentic-apis. This repo contains the toolchain-apis.

## Installation

You can install this repository as a [package](https://pypi.org/project/llama-toolchain/) with `pip install llama-toolchain`

If you want to install from source:

```bash
mkdir -p ~/local
cd ~/local
git clone git@github.com:meta-llama/llama-toolchain.git

conda create -n toolchain python=3.10
conda activate toolchain

cd llama-toolchain
pip install -e .
```

## The Llama CLI

The `llama` CLI makes it easy to configure and run the Llama toolchain. Read the [CLI reference](docs/cli_reference.md) for details.
