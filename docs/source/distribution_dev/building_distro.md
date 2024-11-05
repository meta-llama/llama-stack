# Developer Guide: Assemble a Llama Stack Distribution

> NOTE: This doc may be out-of-date.

This guide will walk you through the steps to get started with building a Llama Stack distributiom from scratch with your choice of API providers. Please see the [Getting Started Guide](./getting_started.md) if you just want the basic steps to start a Llama Stack distribution.

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

```
+------------------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| Template Name                | Providers                                  | Description                                                                      |
+------------------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| bedrock                      | {                                          | Use Amazon Bedrock APIs.                                                         |
|                              |   "inference": "remote::bedrock",          |                                                                                  |
|                              |   "memory": "meta-reference",              |                                                                                  |
|                              |   "safety": "meta-reference",              |                                                                                  |
|                              |   "agents": "meta-reference",              |                                                                                  |
|                              |   "telemetry": "meta-reference"            |                                                                                  |
|                              | }                                          |                                                                                  |
+------------------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| databricks                   | {                                          | Use Databricks for running LLM inference                                         |
|                              |   "inference": "remote::databricks",       |                                                                                  |
|                              |   "memory": "meta-reference",              |                                                                                  |
|                              |   "safety": "meta-reference",              |                                                                                  |
|                              |   "agents": "meta-reference",              |                                                                                  |
|                              |   "telemetry": "meta-reference"            |                                                                                  |
|                              | }                                          |                                                                                  |
+------------------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| fireworks                    | {                                          | Use Fireworks.ai for running LLM inference                                       |
|                              |   "inference": "remote::fireworks",        |                                                                                  |
|                              |   "memory": [                              |                                                                                  |
|                              |     "meta-reference",                      |                                                                                  |
|                              |     "remote::weaviate",                    |                                                                                  |
|                              |     "remote::chromadb",                    |                                                                                  |
|                              |     "remote::pgvector"                     |                                                                                  |
|                              |   ],                                       |                                                                                  |
|                              |   "safety": "meta-reference",              |                                                                                  |
|                              |   "agents": "meta-reference",              |                                                                                  |
|                              |   "telemetry": "meta-reference"            |                                                                                  |
|                              | }                                          |                                                                                  |
+------------------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| hf-endpoint                  | {                                          | Like local, but use Hugging Face Inference Endpoints for running LLM inference.  |
|                              |   "inference": "remote::hf::endpoint",     | See https://hf.co/docs/api-endpoints.                                            |
|                              |   "memory": "meta-reference",              |                                                                                  |
|                              |   "safety": "meta-reference",              |                                                                                  |
|                              |   "agents": "meta-reference",              |                                                                                  |
|                              |   "telemetry": "meta-reference"            |                                                                                  |
|                              | }                                          |                                                                                  |
+------------------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| hf-serverless                | {                                          | Like local, but use Hugging Face Inference API (serverless) for running LLM      |
|                              |   "inference": "remote::hf::serverless",   | inference.                                                                       |
|                              |   "memory": "meta-reference",              | See https://hf.co/docs/api-inference.                                            |
|                              |   "safety": "meta-reference",              |                                                                                  |
|                              |   "agents": "meta-reference",              |                                                                                  |
|                              |   "telemetry": "meta-reference"            |                                                                                  |
|                              | }                                          |                                                                                  |
+------------------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| meta-reference-gpu           | {                                          | Use code from `llama_stack` itself to serve all llama stack APIs                 |
|                              |   "inference": "meta-reference",           |                                                                                  |
|                              |   "memory": [                              |                                                                                  |
|                              |     "meta-reference",                      |                                                                                  |
|                              |     "remote::chromadb",                    |                                                                                  |
|                              |     "remote::pgvector"                     |                                                                                  |
|                              |   ],                                       |                                                                                  |
|                              |   "safety": "meta-reference",              |                                                                                  |
|                              |   "agents": "meta-reference",              |                                                                                  |
|                              |   "telemetry": "meta-reference"            |                                                                                  |
|                              | }                                          |                                                                                  |
+------------------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| meta-reference-quantized-gpu | {                                          | Use code from `llama_stack` itself to serve all llama stack APIs                 |
|                              |   "inference": "meta-reference-quantized", |                                                                                  |
|                              |   "memory": [                              |                                                                                  |
|                              |     "meta-reference",                      |                                                                                  |
|                              |     "remote::chromadb",                    |                                                                                  |
|                              |     "remote::pgvector"                     |                                                                                  |
|                              |   ],                                       |                                                                                  |
|                              |   "safety": "meta-reference",              |                                                                                  |
|                              |   "agents": "meta-reference",              |                                                                                  |
|                              |   "telemetry": "meta-reference"            |                                                                                  |
|                              | }                                          |                                                                                  |
+------------------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| ollama                       | {                                          | Use ollama for running LLM inference                                             |
|                              |   "inference": "remote::ollama",           |                                                                                  |
|                              |   "memory": [                              |                                                                                  |
|                              |     "meta-reference",                      |                                                                                  |
|                              |     "remote::chromadb",                    |                                                                                  |
|                              |     "remote::pgvector"                     |                                                                                  |
|                              |   ],                                       |                                                                                  |
|                              |   "safety": "meta-reference",              |                                                                                  |
|                              |   "agents": "meta-reference",              |                                                                                  |
|                              |   "telemetry": "meta-reference"            |                                                                                  |
|                              | }                                          |                                                                                  |
+------------------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| tgi                          | {                                          | Use TGI for running LLM inference                                                |
|                              |   "inference": "remote::tgi",              |                                                                                  |
|                              |   "memory": [                              |                                                                                  |
|                              |     "meta-reference",                      |                                                                                  |
|                              |     "remote::chromadb",                    |                                                                                  |
|                              |     "remote::pgvector"                     |                                                                                  |
|                              |   ],                                       |                                                                                  |
|                              |   "safety": "meta-reference",              |                                                                                  |
|                              |   "agents": "meta-reference",              |                                                                                  |
|                              |   "telemetry": "meta-reference"            |                                                                                  |
|                              | }                                          |                                                                                  |
+------------------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| together                     | {                                          | Use Together.ai for running LLM inference                                        |
|                              |   "inference": "remote::together",         |                                                                                  |
|                              |   "memory": [                              |                                                                                  |
|                              |     "meta-reference",                      |                                                                                  |
|                              |     "remote::weaviate"                     |                                                                                  |
|                              |   ],                                       |                                                                                  |
|                              |   "safety": "remote::together",            |                                                                                  |
|                              |   "agents": "meta-reference",              |                                                                                  |
|                              |   "telemetry": "meta-reference"            |                                                                                  |
|                              | }                                          |                                                                                  |
+------------------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| vllm                         | {                                          | Like local, but use vLLM for running LLM inference                               |
|                              |   "inference": "vllm",                     |                                                                                  |
|                              |   "memory": "meta-reference",              |                                                                                  |
|                              |   "safety": "meta-reference",              |                                                                                  |
|                              |   "agents": "meta-reference",              |                                                                                  |
|                              |   "telemetry": "meta-reference"            |                                                                                  |
|                              | }                                          |                                                                                  |
+------------------------------+--------------------------------------------+----------------------------------------------------------------------------------+
```

You may then pick a template to build your distribution with providers fitted to your liking.

```
llama stack build --template tgi
```

```
$ llama stack build --template tgi
...
...
You can now edit ~/meta-llama/llama-stack/tmp/configs/tgi-run.yaml and run `llama stack run ~/meta-llama/llama-stack/tmp/configs/tgi-run.yaml`
```

#### Building from config file
- In addition to templates, you may customize the build to your liking through editing config files and build from config files with the following command.

- The config file will be of contents like the ones in `llama_stack/distributions/templates/`.

```
$ cat llama_stack/templates/ollama/build.yaml

name: ollama
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
llama stack build --config llama_stack/templates/ollama/build.yaml
```

#### How to build distribution with Docker image

> [!TIP]
> Podman is supported as an alternative to Docker. Set `DOCKER_BINARY` to `podman` in your environment to use Podman.

To build a docker image, you may start off from a template and use the `--image-type docker` flag to specify `docker` as the build image type.

```
llama stack build --template local --image-type docker
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

After this step is successful, you should be able to find a run configuration spec in `~/.llama/builds/conda/tgi-run.yaml` with the following contents. You may edit this file to change the settings.

As you can see, we did basic configuration above and configured:
- inference to run on model `Meta-Llama3.1-8B-Instruct` (obtained from `llama model list`)
- Llama Guard safety shield with model `Llama-Guard-3-1B`
- Prompt Guard safety shield with model `Prompt-Guard-86M`

For how these configurations are stored as yaml, checkout the file printed at the end of the configuration.

Note that all configurations as well as models are stored in `~/.llama`


## Step 2. Run
Now, let's start the Llama Stack Distribution Server. You will need the YAML configuration file which was written out at the end by the `llama stack build` step.

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
> Configuration is in `~/.llama/builds/local/conda/tgi-run.yaml`. Feel free to increase `max_seq_len`.

> [!IMPORTANT]
> The "local" distribution inference server currently only supports CUDA. It will not work on Apple Silicon machines.

> [!TIP]
> You might need to use the flag `--disable-ipv6` to  Disable IPv6 support

This server is running a Llama model locally.
