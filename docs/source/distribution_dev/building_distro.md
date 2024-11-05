# Developer Guide: Assemble a Llama Stack Distribution


This guide will walk you through the steps to get started with building a Llama Stack distributiom from scratch with your choice of API providers. Please see the [Getting Started Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html) if you just want the basic steps to start a Llama Stack distribution.

## Step 1. Build

```
llama stack build -h

usage: llama stack build [-h] [--config CONFIG] [--template TEMPLATE] [--list-templates | --no-list-templates] [--image-type {conda,docker}]

Build a Llama stack container

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to a config file to use for the build. You can find example configs in llama_stack/distribution/example_configs. If this argument is not provided, you will be prompted to enter information interactively
  --template TEMPLATE   Name of the example template config to use for build. You may use `llama stack build --list-templates` to check out the available templates
  --list-templates, --no-list-templates
                        Show the available templates for building a Llama Stack distribution
  --image-type {conda,docker}
                        Image Type to use for the build. This can be either conda or docker. If not specified, will use the image type from the template config.

```
We will start build our distribution (in the form of a Conda environment, or Docker image). In this step, we will specify:
- `name`: the name for our distribution (e.g. `my-stack`)
- `image_type`: our build image type (`conda | docker`)
- `distribution_spec`: our distribution specs for specifying API providers
  - `description`: a short description of the configurations for the distribution
  - `providers`: specifies the underlying implementation for serving each API endpoint
  - `image_type`: `conda` | `docker` to specify whether to build the distribution in the form of Docker image or Conda environment.

After this step is complete, a file named `<name>-build.yaml` and template file `<name>-run.yaml` will be generated and saved at the output file path specified at the end of the command.


You have 3 options for building your distribution:
1.1 Building from scratch
1.2. Building from a template
1.3. Building from a pre-existing build config file


### 1.1 Building from scratch
- For a new user, we could start off with running `llama stack build` which will allow you to a interactively enter wizard where you will be prompted to enter build configurations.
```
llama stack build

> Enter a name for your Llama Stack (e.g. my-local-stack): my-stack
> Enter the image type you want your Llama Stack to be built as (docker or conda): conda

Llama Stack is composed of several APIs working together. Let's select
the provider types (implementations) you want to use for these APIs.

Tip: use <TAB> to see options for the providers.

> Enter provider for API inference: meta-reference
> Enter provider for API safety: meta-reference
> Enter provider for API agents: meta-reference
> Enter provider for API memory: meta-reference
> Enter provider for API datasetio: meta-reference
> Enter provider for API scoring: meta-reference
> Enter provider for API eval: meta-reference
> Enter provider for API telemetry: meta-reference

 > (Optional) Enter a short description for your Llama Stack:

You can now edit ~/.llama/distributions/llamastack-my-local-stack/my-local-stack-run.yaml and run `llama stack run ~/.llama/distributions/llamastack-my-local-stack/my-local-stack-run.yaml`
```

### 1.2 Building from a template
- To build from alternative API providers, we provide distribution templates for users to get started building a distribution backed by different providers.

The following command will allow you to see the available templates and their corresponding providers.
```
llama stack build --list-templates
```

```
+------------------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| Template Name                | Providers                                  | Description                                                                      |
+------------------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| hf-serverless                | {                                          | Like local, but use Hugging Face Inference API (serverless) for running LLM      |
|                              |   "inference": "remote::hf::serverless",   | inference.                                                                       |
|                              |   "memory": "meta-reference",              | See https://hf.co/docs/api-inference.                                            |
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
| databricks                   | {                                          | Use Databricks for running LLM inference                                         |
|                              |   "inference": "remote::databricks",       |                                                                                  |
|                              |   "memory": "meta-reference",              |                                                                                  |
|                              |   "safety": "meta-reference",              |                                                                                  |
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
| bedrock                      | {                                          | Use Amazon Bedrock APIs.                                                         |
|                              |   "inference": "remote::bedrock",          |                                                                                  |
|                              |   "memory": "meta-reference",              |                                                                                  |
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
| hf-endpoint                  | {                                          | Like local, but use Hugging Face Inference Endpoints for running LLM inference.  |
|                              |   "inference": "remote::hf::endpoint",     | See https://hf.co/docs/api-endpoints.                                            |
|                              |   "memory": "meta-reference",              |                                                                                  |
|                              |   "safety": "meta-reference",              |                                                                                  |
|                              |   "agents": "meta-reference",              |                                                                                  |
|                              |   "telemetry": "meta-reference"            |                                                                                  |
|                              | }                                          |                                                                                  |
+------------------------------+--------------------------------------------+----------------------------------------------------------------------------------+
```

You may then pick a template to build your distribution with providers fitted to your liking.

For example, to build a distribution with TGI as the inference provider, you can run:
```
llama stack build --template tgi
```

```
$ llama stack build --template tgi
...
You can now edit ~/.llama/distributions/llamastack-tgi/tgi-run.yaml and run `llama stack run ~/.llama/distributions/llamastack-tgi/tgi-run.yaml`
```

### 1.3 Building from a pre-existing build config file
- In addition to templates, you may customize the build to your liking through editing config files and build from config files with the following command.

- The config file will be of contents like the ones in `llama_stack/templates/*build.yaml`.

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

### How to build distribution with Docker image
> [!TIP]
> Podman is supported as an alternative to Docker. Set `DOCKER_BINARY` to `podman` in your environment to use Podman.

To build a docker image, you may start off from a template and use the `--image-type docker` flag to specify `docker` as the build image type.

```
llama stack build --template ollama --image-type docker
```

```
$ llama stack build --template ollama --image-type docker
...
Dockerfile created successfully in /tmp/tmp.viA3a3Rdsg/DockerfileFROM python:3.10-slim
...

You can now edit ~/meta-llama/llama-stack/tmp/configs/ollama-run.yaml and run `llama stack run ~/meta-llama/llama-stack/tmp/configs/ollama-run.yaml`
```

After this step is successful, you should be able to find the built docker image and test it with `llama stack run <path/to/run.yaml>`. 


## Step 2. Run
Now, let's start the Llama Stack Distribution Server. You will need the YAML configuration file which was written out at the end by the `llama stack build` step.

```
llama stack run ~/.llama/distributions/llamastack-my-local-stack/my-local-stack-run.yaml
```

```
$ llama stack run ~/.llama/distributions/llamastack-my-local-stack/my-local-stack-run.yaml

Loaded model...
Serving API datasets
 GET /datasets/get
 GET /datasets/list
 POST /datasets/register
Serving API inspect
 GET /health
 GET /providers/list
 GET /routes/list
Serving API inference
 POST /inference/chat_completion
 POST /inference/completion
 POST /inference/embeddings
Serving API scoring_functions
 GET /scoring_functions/get
 GET /scoring_functions/list
 POST /scoring_functions/register
Serving API scoring
 POST /scoring/score
 POST /scoring/score_batch
Serving API memory_banks
 GET /memory_banks/get
 GET /memory_banks/list
 POST /memory_banks/register
Serving API memory
 POST /memory/insert
 POST /memory/query
Serving API safety
 POST /safety/run_shield
Serving API eval
 POST /eval/evaluate
 POST /eval/evaluate_batch
 POST /eval/job/cancel
 GET /eval/job/result
 GET /eval/job/status
Serving API shields
 GET /shields/get
 GET /shields/list
 POST /shields/register
Serving API datasetio
 GET /datasetio/get_rows_paginated
Serving API telemetry
 GET /telemetry/get_trace
 POST /telemetry/log_event
Serving API models
 GET /models/get
 GET /models/list
 POST /models/register
Serving API agents
 POST /agents/create
 POST /agents/session/create
 POST /agents/turn/create
 POST /agents/delete
 POST /agents/session/delete
 POST /agents/session/get
 POST /agents/step/get
 POST /agents/turn/get

Listening on ['::', '0.0.0.0']:5000
INFO:     Started server process [2935911]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://['::', '0.0.0.0']:5000 (Press CTRL+C to quit)
INFO:     2401:db00:35c:2d2b:face:0:c9:0:54678 - "GET /models/list HTTP/1.1" 200 OK
```

> [!IMPORTANT]
> The "local" distribution inference server currently only supports CUDA. It will not work on Apple Silicon machines.

> [!TIP]
> You might need to use the flag `--disable-ipv6` to  Disable IPv6 support
