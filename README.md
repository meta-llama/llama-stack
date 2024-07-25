# llama-toolchain

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

## Appendix: Running FP8

If you want to run FP8, you need the `fbgemm-gpu` package which requires `torch >= 2.4.0` (currently only in nightly, but releasing shortly...)

```bash
ENV=fp8_env
conda create -n $ENV python=3.10
conda activate $ENV

pip3 install -r fp8_requirements.txt
```
