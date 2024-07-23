# llama-toolchain

This repo contains the API specifications for various components of the Llama Stack as well implementations for some of those APIs like model inference.
The Stack consists of toolchain-apis and agentic-apis. This repo contains the toolchain-apis

## Installation

You can install this repository as a [package](https://pypi.org/project/llama-toolchain/) by just doing `pip install llama-toolchain`

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

## Test with cli

We have built a llama cli to make it easy to configure / run parts of the toolchain
```
llama --help

usage: llama [-h] {download,inference,model,agentic_system} ...

Welcome to the LLama cli

options:
  -h, --help            show this help message and exit

subcommands:
  {download,inference,model,agentic_system}
```
There are several subcommands to help get you started

## Start inference server that can run the llama models
```bash
llama inference configure
llama inference start
```


## Test client
```bash
python -m llama_toolchain.inference.client localhost 5000

Initializing client for http://localhost:5000
User>hello world, help me out here
Assistant> Hello! I'd be delighted to help you out. What's on your mind? Do you have a question, a problem, or just need someone to chat with? I'm all ears!
```


## Running FP8

You need `fbgemm-gpu` package which requires torch >= 2.4.0 (currently only in nightly, but releasing shortly...).

```bash
ENV=fp8_env
conda create -n $ENV python=3.10
conda activate $ENV

pip3 install -r fp8_requirements.txt
```
