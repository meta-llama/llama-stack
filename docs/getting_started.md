# llama-stack

[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-stack)](https://pypi.org/project/llama-stack/)
[![Discord](https://img.shields.io/discord/1257833999603335178)](https://discord.gg/TZAAYNVtrU)

This repository contains the specifications and implementations of the APIs which are part of the Llama Stack.

The Llama Stack defines and standardizes the building blocks needed to bring generative AI applications to market. These blocks span the entire development lifecycle: from model training and fine-tuning, through product evaluation, to invoking AI agents in production. Beyond definition, we're developing open-source versions and partnering with cloud providers, ensuring developers can assemble AI solutions using consistent, interlocking pieces across platforms. The ultimate goal is to accelerate innovation in the AI space.

The Stack APIs are rapidly improving, but still very much work in progress and we invite feedback as well as direct contributions.


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

As an example, for Inference, we could have the implementation be backed by open source libraries like `[ torch | vLLM | TensorRT ]` as possible options.

A provider can also be just a pointer to a remote REST service -- for example, cloud providers or dedicated inference providers could serve these APIs.


## Llama Stack Distribution

A Distribution is where APIs and Providers are assembled together to provide a consistent whole to the end application developer. You can mix-and-match providers -- some could be backed by local code and some could be remote. As a hobbyist, you can serve a small model locally, but can choose a cloud provider for a large model. Regardless, the higher level APIs your app needs to work with don't need to change at all. You can even imagine moving across the server / mobile-device boundary as well always using the same uniform set of APIs for developing Generative AI applications.


## Installation

You can install this repository as a [package](https://pypi.org/project/llama-stack/) with `pip install llama-stack`

If you want to install from source:

```bash
mkdir -p ~/local
cd ~/local
git clone git@github.com:meta-llama/llama-stack.git

conda create -n stack python=3.10
conda activate stack

cd llama-stack
$CONDA_PREFIX/bin/pip install -e .
```

# Getting Started

The `llama` CLI tool helps you setup and use the Llama toolchain & agentic systems. It should be available on your path after installing the `llama-stack` package.

This guides allows you to quickly get started with building and running a Llama Stack server in < 5 minutes!

You may also checkout this [notebook](https://github.com/meta-llama/llama-stack/blob/main/docs/getting_started.ipynb) for trying out out demo scripts.

## Quick Cheatsheet
- Quick 3 line command to build and start a LlamaStack server using our Meta Reference implementation for all API endpoints with `conda` as build type.

**`llama stack build`**
- You'll be prompted to enter build information interactively.
```
llama stack build

> Enter an unique name for identifying your Llama Stack build distribution (e.g. my-local-stack): my-local-stack
> Enter the image type you want your distribution to be built with (docker or conda): conda

 Llama Stack is composed of several APIs working together. Let's configure the providers (implementations) you want to use for these APIs.
> Enter the API provider for the inference API: (default=meta-reference): meta-reference
> Enter the API provider for the safety API: (default=meta-reference): meta-reference
> Enter the API provider for the agents API: (default=meta-reference): meta-reference
> Enter the API provider for the memory API: (default=meta-reference): meta-reference
> Enter the API provider for the telemetry API: (default=meta-reference): meta-reference

 > (Optional) Enter a short description for your Llama Stack distribution:

Build spec configuration saved at ~/.conda/envs/llamastack-my-local-stack/my-local-stack-build.yaml
You can now run `llama stack configure my-local-stack`
```

**`llama stack configure`**
- Run `llama stack configure <name>` with the name you have previously defined in `build` step.
```
llama stack configure <name>
```
- You will be prompted to enter configurations for your Llama Stack

```
$ llama stack configure my-local-stack

Could not find my-local-stack. Trying conda build name instead...
Configuring API `inference`...
=== Configuring provider `meta-reference` for API inference...
Enter value for model (default: Llama3.1-8B-Instruct) (required):
Do you want to configure quantization? (y/n): n
Enter value for torch_seed (optional):
Enter value for max_seq_len (default: 4096) (required):
Enter value for max_batch_size (default: 1) (required):

Configuring API `safety`...
=== Configuring provider `meta-reference` for API safety...
Do you want to configure llama_guard_shield? (y/n): n
Do you want to configure prompt_guard_shield? (y/n): n

Configuring API `agents`...
=== Configuring provider `meta-reference` for API agents...
Enter `type` for persistence_store (options: redis, sqlite, postgres) (default: sqlite):

Configuring SqliteKVStoreConfig:
Enter value for namespace (optional):
Enter value for db_path (default: /home/xiyan/.llama/runtime/kvstore.db) (required):

Configuring API `memory`...
=== Configuring provider `meta-reference` for API memory...
> Please enter the supported memory bank type your provider has for memory: vector

Configuring API `telemetry`...
=== Configuring provider `meta-reference` for API telemetry...

> YAML configuration has been written to ~/.llama/builds/conda/my-local-stack-run.yaml.
You can now run `llama stack run my-local-stack --port PORT`
```

**`llama stack run`**
- Run `llama stack run <name>` with the name you have previously defined.
```
llama stack run my-local-stack

...
> initializing model parallel with size 1
> initializing ddp with size 1
> initializing pipeline with size 1
...
Finished model load YES READY
Serving POST /inference/chat_completion
Serving POST /inference/completion
Serving POST /inference/embeddings
Serving POST /memory_banks/create
Serving DELETE /memory_bank/documents/delete
Serving DELETE /memory_banks/drop
Serving GET /memory_bank/documents/get
Serving GET /memory_banks/get
Serving POST /memory_bank/insert
Serving GET /memory_banks/list
Serving POST /memory_bank/query
Serving POST /memory_bank/update
Serving POST /safety/run_shield
Serving POST /agentic_system/create
Serving POST /agentic_system/session/create
Serving POST /agentic_system/turn/create
Serving POST /agentic_system/delete
Serving POST /agentic_system/session/delete
Serving POST /agentic_system/session/get
Serving POST /agentic_system/step/get
Serving POST /agentic_system/turn/get
Serving GET /telemetry/get_trace
Serving POST /telemetry/log_event
Listening on :::5000
INFO:     Started server process [587053]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://[::]:5000 (Press CTRL+C to quit)
```


## Step 1. Build
In the following steps, imagine we'll be working with a `Meta-Llama3.1-8B-Instruct` model. We will name our build `8b-instruct` to help us remember the config. We will start build our distribution (in the form of a Conda environment, or Docker image). In this step, we will specify:
- `name`: the name for our distribution (e.g. `8b-instruct`)
- `image_type`: our build image type (`conda | docker`)
- `distribution_spec`: our distribution specs for specifying API providers
  - `description`: a short description of the configurations for the distribution
  - `providers`: specifies the underlying implementation for serving each API endpoint
  - `image_type`: `conda` | `docker` to specify whether to build the distribution in the form of Docker image or Conda environment.


At the end of build command, we will generate `<name>-build.yaml` file storing the build configurations.

After this step is complete, a file named `<name>-build.yaml` will be generated and saved at the output file path specified at the end of the command.

#### Building from scratch
- For a new user, we could start off with running `llama stack build` which will allow you to a interactively enter wizard where you will be prompted to enter build configurations.
```
llama stack build
```

Running the command above will allow you to fill in the configuration to build your Llama Stack distribution, you will see the following outputs.

```
> Enter an unique name for identifying your Llama Stack build distribution (e.g. my-local-stack): 8b-instruct
> Enter the image type you want your distribution to be built with (docker or conda): conda

 Llama Stack is composed of several APIs working together. Let's configure the providers (implementations) you want to use for these APIs.
> Enter the API provider for the inference API: (default=meta-reference): meta-reference
> Enter the API provider for the safety API: (default=meta-reference): meta-reference
> Enter the API provider for the agents API: (default=meta-reference): meta-reference
> Enter the API provider for the memory API: (default=meta-reference): meta-reference
> Enter the API provider for the telemetry API: (default=meta-reference): meta-reference

 > (Optional) Enter a short description for your Llama Stack distribution:

Build spec configuration saved at ~/.conda/envs/llamastack-my-local-llama-stack/8b-instruct-build.yaml
```

**Ollama (optional)**

If you plan to use Ollama for inference, you'll need to install the server [via these instructions](https://ollama.com/download).


#### Building from templates
- To build from alternative API providers, we provide distribution templates for users to get started building a distribution backed by different providers.

The following command will allow you to see the available templates and their corresponding providers.
```
llama stack build --list-templates
```

![alt text](resources/list-templates.png)

You may then pick a template to build your distribution with providers fitted to your liking.

```
llama stack build --template local-tgi --name my-tgi-stack
```

```
$ llama stack build --template local-tgi --name my-tgi-stack
...
...
Build spec configuration saved at ~/.conda/envs/llamastack-my-tgi-stack/my-tgi-stack-build.yaml
You may now run `llama stack configure my-tgi-stack` or `llama stack configure ~/.conda/envs/llamastack-my-tgi-stack/my-tgi-stack-build.yaml`
```

#### Building from config file
- In addition to templates, you may customize the build to your liking through editing config files and build from config files with the following command.

- The config file will be of contents like the ones in `llama_stack/distributions/templates/`.

```
$ cat llama_stack/distribution/templates/local-ollama-build.yaml

name: local-ollama
distribution_spec:
  description: Like local, but use ollama for running LLM inference
  providers:
    inference: remote::ollama
    memory: meta-reference
    safety: meta-reference
    agents: meta-reference
    telemetry: meta-reference
image_type: conda
```

```
llama stack build --config llama_stack/distribution/templates/local-ollama-build.yaml
```

#### How to build distribution with Docker image

> [!TIP]
> Podman is supported as an alternative to Docker. Set `DOCKER_BINARY` to `podman` in your environment to use Podman.

To build a docker image, you may start off from a template and use the `--image-type docker` flag to specify `docker` as the build image type.

```
llama stack build --template local --image-type docker --name docker-0
```

Alternatively, you may use a config file and set `image_type` to `docker` in our `<name>-build.yaml` file, and run `llama stack build <name>-build.yaml`. The `<name>-build.yaml` will be of contents like:

```
name: local-docker-example
distribution_spec:
  description: Use code from `llama_stack` itself to serve all llama stack APIs
  docker_image: null
  providers:
    inference: meta-reference
    memory: meta-reference-faiss
    safety: meta-reference
    agentic_system: meta-reference
    telemetry: console
image_type: docker
```

The following command allows you to build a Docker image with the name `<name>`
```
llama stack build --config <name>-build.yaml

Dockerfile created successfully in /tmp/tmp.I0ifS2c46A/DockerfileFROM python:3.10-slim
WORKDIR /app
...
...
You can run it with: podman run -p 8000:8000 llamastack-docker-local
Build spec configuration saved at ~/.llama/distributions/docker/docker-local-build.yaml
```


## Step 2. Configure
After our distribution is built (either in form of docker or conda environment), we will run the following command to
```
llama stack configure [ <name> | <docker-image-name> | <path/to/name.build.yaml>]
```
- For `conda` environments: <path/to/name.build.yaml> would be the generated build spec saved from Step 1.
- For `docker` images downloaded from Dockerhub, you could also use <docker-image-name> as the argument.
   - Run `docker images` to check list of available images on your machine.

```
$ llama stack configure 8b-instruct

Configuring API: inference (meta-reference)
Enter value for model (existing: Meta-Llama3.1-8B-Instruct) (required):
Enter value for quantization (optional):
Enter value for torch_seed (optional):
Enter value for max_seq_len (existing: 4096) (required):
Enter value for max_batch_size (existing: 1) (required):

Configuring API: memory (meta-reference-faiss)

Configuring API: safety (meta-reference)
Do you want to configure llama_guard_shield? (y/n): y
Entering sub-configuration for llama_guard_shield:
Enter value for model (default: Llama-Guard-3-8B) (required):
Enter value for excluded_categories (default: []) (required):
Enter value for disable_input_check (default: False) (required):
Enter value for disable_output_check (default: False) (required):
Do you want to configure prompt_guard_shield? (y/n): y
Entering sub-configuration for prompt_guard_shield:
Enter value for model (default: Prompt-Guard-86M) (required):

Configuring API: agentic_system (meta-reference)
Enter value for brave_search_api_key (optional):
Enter value for bing_search_api_key (optional):
Enter value for wolfram_api_key (optional):

Configuring API: telemetry (console)

YAML configuration has been written to ~/.llama/builds/conda/8b-instruct-run.yaml
```

After this step is successful, you should be able to find a run configuration spec in `~/.llama/builds/conda/8b-instruct-run.yaml` with the following contents. You may edit this file to change the settings.

As you can see, we did basic configuration above and configured:
- inference to run on model `Meta-Llama3.1-8B-Instruct` (obtained from `llama model list`)
- Llama Guard safety shield with model `Llama-Guard-3-8B`
- Prompt Guard safety shield with model `Prompt-Guard-86M`

For how these configurations are stored as yaml, checkout the file printed at the end of the configuration.

Note that all configurations as well as models are stored in `~/.llama`


## Step 3. Run
Now, let's start the Llama Stack Distribution Server. You will need the YAML configuration file which was written out at the end by the `llama stack configure` step.

```
llama stack run 8b-instruct
```

You should see the Llama Stack server start and print the APIs that it is supporting

```
$ llama stack run 8b-instruct

> initializing model parallel with size 1
> initializing ddp with size 1
> initializing pipeline with size 1
Loaded in 19.28 seconds
NCCL version 2.20.5+cuda12.4
Finished model load YES READY
Serving POST /inference/batch_chat_completion
Serving POST /inference/batch_completion
Serving POST /inference/chat_completion
Serving POST /inference/completion
Serving POST /safety/run_shield
Serving POST /agentic_system/memory_bank/attach
Serving POST /agentic_system/create
Serving POST /agentic_system/session/create
Serving POST /agentic_system/turn/create
Serving POST /agentic_system/delete
Serving POST /agentic_system/session/delete
Serving POST /agentic_system/memory_bank/detach
Serving POST /agentic_system/session/get
Serving POST /agentic_system/step/get
Serving POST /agentic_system/turn/get
Listening on :::5000
INFO:     Started server process [453333]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://[::]:5000 (Press CTRL+C to quit)
```

> [!NOTE]
> Configuration is in `~/.llama/builds/local/conda/8b-instruct-run.yaml`. Feel free to increase `max_seq_len`.

> [!IMPORTANT]
> The "local" distribution inference server currently only supports CUDA. It will not work on Apple Silicon machines.

> [!TIP]
> You might need to use the flag `--disable-ipv6` to  Disable IPv6 support

This server is running a Llama model locally.

## Step 4. Test with Client
Once the server is setup, we can test it with a client to see the example outputs.
```
cd /path/to/llama-stack
conda activate <env>  # any environment containing the llama-stack pip package will work

python -m llama_stack.apis.inference.client localhost 5000
```

This will run the chat completion client and query the distribution’s /inference/chat_completion API.

Here is an example output:
```
User>hello world, write me a 2 sentence poem about the moon
Assistant> Here's a 2-sentence poem about the moon:

The moon glows softly in the midnight sky,
A beacon of wonder, as it passes by.
```

Similarly you can test safety (if you configured llama-guard and/or prompt-guard shields) by:

```
python -m llama_stack.apis.safety.client localhost 5000
```

You can find more example scripts with client SDKs to talk with the Llama Stack server in our [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repo.
