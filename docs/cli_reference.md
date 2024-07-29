# Llama CLI Reference

The `llama` CLI tool helps you setup and use the Llama toolchain & agentic systems. It should be available on your path after installing the `llama-toolchain` package.

```
$ llama --help

Welcome to the Llama CLI

Usage: llama [-h] {download,inference,model} ...


Options:
  -h, --help            Show this help message and exit


Subcommands:
  {download,inference,model}
```

## Step 1. Get the models

First, you need models locally. You can get the models from [HuggingFace](https://huggingface.co/meta-llama) or [directly from Meta](https://llama.meta.com/llama-downloads/). The download command streamlines the process.


```
$ llama download --help
usage: llama download [-h] [--hf-token HF_TOKEN] [--ignore-patterns IGNORE_PATTERNS] repo_id

Download a model from the Hugging Face Hub

positional arguments:
  repo_id               Name of the repository on Hugging Face Hub eg. llhf/Meta-Llama-3.1-70B-Instruct

options:
  -h, --help            show this help message and exit
  --hf-token HF_TOKEN   Hugging Face API token. Needed for gated models like Llama2. Will also try to read environment variable `HF_TOKEN` as default.
  --ignore-patterns IGNORE_PATTERNS
                        If provided, files matching any of the patterns are not downloaded. Defaults to ignoring safetensors files to avoid downloading duplicate weights.

# Here are some examples on how to use this command:

llama download --repo-id meta-llama/Llama-2-7b-hf --hf-token <HF_TOKEN>
llama download --repo-id meta-llama/Llama-2-7b-hf --output-dir /data/my_custom_dir --hf-token <HF_TOKEN>
HF_TOKEN=<HF_TOKEN> llama download --repo-id meta-llama/Llama-2-7b-hf

The output directory will be used to load models and tokenizers for inference.
```

1. Create and get a Hugging Face access token [here](https://huggingface.co/settings/tokens)
2. Set the `HF_TOKEN` environment variable

```
export HF_TOKEN=YOUR_TOKEN_HERE
llama download meta-llama/Meta-Llama-3.1-70B-Instruct
```

## Step 2: Understand the models
The `llama model` command helps you explore the model’s interface.

```
$ llama model --help
usage: llama model [-h] {template} ...


Describe llama model interfaces


options:
  -h, --help  show this help message and exit


model_subcommands:
  {template}


Example: llama model <subcommand> <options>
```

You can run `llama model template` see all of the templates and their tokens:


```
$ llama model template


system-message-builtin-and-custom-tools
system-message-builtin-tools-only
system-message-custom-tools-only
system-message-default
assistant-message-builtin-tool-call
assistant-message-custom-tool-call
assistant-message-default
tool-message-failure
tool-message-success
user-message-default
```

And fetch an example by passing it to `--template`:

```
llama model template --template tool-message-success


llama model template --template tool-message-success
<|start_header_id|>ipython<|end_header_id|>


completed
[stdout]{"results":["something something"]}[/stdout]<|eot_id|>
```

## Step 3. Start the inference server

Once you have a model, the magic begins with inference. The `llama inference` command can help you configure and launch the Llama Stack inference server.

```
$ llama inference --help


usage: llama inference [-h] {start,configure} ...


Run inference on a llama model


options:
  -h, --help         show this help message and exit


inference_subcommands:
  {start,configure}


Example: llama inference start <options>
```

Run `llama inference configure` to setup your configuration at `~/.llama/configs/inference.yaml`. You’ll set up variables like:


* the directory where you stored the models you downloaded from step 1
* the model parallel size (1 for 8B models, 8 for 70B/405B)


Once you’ve configured the inference server, run `llama inference start`. The model will load into GPU and you’ll be able to send requests once you see the server ready.


If you want to use a different model, re-run `llama inference configure` to update the model path and llama inference start to start again.


Run `llama inference --help` for more information.


## Step 4. Start the agentic system

The `llama agentic_system` command sets up the configuration file the agentic client code expects.

For example, let’s run the included chat app:

```
llama agentic_system configure
mesop app/main.py
```

For more information run `llama agentic_system --help`.
