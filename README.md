# llama-stack

[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-toolchain)](https://pypi.org/project/llama-toolchain/)
[![Discord](https://img.shields.io/discord/1257833999603335178)](https://discord.gg/TZAAYNVtrU)

This repository contains the specifications and implementations of the APIs which are part of the Llama Stack.

The [Llama Stack](https://github.com/meta-llama/llama-toolchain/pull/8) defines and standardizes the building blocks needed to bring generative AI applications to market. These blocks span the entire development lifecycle: from model training and fine-tuning, through product evaluation, to invoking AI agents in production. Beyond definition, we're developing open-source versions and partnering with cloud providers, ensuring developers can assemble AI solutions using consistent, interlocking pieces across platforms. The ultimate goal is to accelerate innovation in the AI space.

The Stack APIs are rapidly improving, but still very much Work in Progress and we invite feedback as well as direct contributions.


## APIs

The Llama Stack consists of the following set of APIs:

- Inference
- Safety
- Memory
- Agentic System
- Evaluation
- Post Training
- Synthetic Data Generation
- Reward Scoring

Each of the APIs themselves is a collection of REST endpoints.


## API Providers

A Provider is what makes the API real -- they provide the actual implementation backing the API.

As an example, for Inference, we could have the implementation be backed by primitives from `[ torch | vLLM | TensorRT ]` as possible options.

A provider can also be just a pointer to a remote REST service -- for example, cloud providers like `[ aws | gcp ]` could possibly serve these APIs.


## Llama Stack Distribution

A Distribution is where APIs and Providers are assembled together to provide a consistent whole to the end application developer. You can mix-and-match providers -- some could be backed by local code and some could be remote. As a hobbyist, you can serve a small model locally, but can choose a cloud provider for a large model. Regardless, the higher level APIs your app needs to work with don't need to change at all. You can even imagine moving across the server / mobile-device boundary as well always using the same uniform set of APIs for developing Generative AI applications.


## Installation

You can install this repository as a [package](https://pypi.org/project/llama-toolchain/) with `pip install llama-toolchain`

If you want to install from source:

```bash
mkdir -p ~/local
cd ~/local
git clone git@github.com:meta-llama/llama-stack.git

conda create -n stack python=3.10
conda activate stack

cd llama-stack
pip install -e .
```

## The Llama CLI

The `llama` CLI makes it easy to work with the Llama Stack set of tools, including installing and running Distributions, downloading models, studying model prompt formats, etc. Please see the [CLI reference](docs/cli_reference.md) for details.
