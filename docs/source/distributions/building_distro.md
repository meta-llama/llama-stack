# Build your own Distribution


This guide will walk you through the steps to get started with building a Llama Stack distribution from scratch with your choice of API providers.


## Llama Stack Build

In order to build your own distribution, we recommend you clone the `llama-stack` repository.


```
git clone git@github.com:meta-llama/llama-stack.git
cd llama-stack
pip install -e .

llama stack build -h
```

We will start build our distribution (in the form of a Conda environment, or Docker image). In this step, we will specify:
- `name`: the name for our distribution (e.g. `my-stack`)
- `image_type`: our build image type (`conda | docker`)
- `platform`: the platform for docker image (e.g. `linux/arm64`)
- `distribution_spec`: our distribution specs for specifying API providers
  - `description`: a short description of the configurations for the distribution
  - `providers`: specifies the underlying implementation for serving each API endpoint
  - `image_type`: `conda` | `docker` to specify whether to build the distribution in the form of Docker image or Conda environment.

After this step is complete, a file named `<name>-build.yaml` and template file `<name>-run.yaml` will be generated and saved at the output file path specified at the end of the command.

::::{tab-set}
:::{tab-item} Building from Scratch

- For a new user, we could start off with running `llama stack build` which will allow you to a interactively enter wizard where you will be prompted to enter build configurations.
```
llama stack build

> Enter a name for your Llama Stack (e.g. my-local-stack): my-stack
> Enter the image type you want your Llama Stack to be built as (docker or conda): conda
> Enter the target platform you want your Llama Stack to be built for:

Llama Stack is composed of several APIs working together. Let's select
the provider types (implementations) you want to use for these APIs.

Tip: use <TAB> to see options for the providers.

> Enter provider for API inference: inline::meta-reference
> Enter provider for API safety: inline::llama-guard
> Enter provider for API agents: inline::meta-reference
> Enter provider for API memory: inline::faiss
> Enter provider for API datasetio: inline::meta-reference
> Enter provider for API scoring: inline::meta-reference
> Enter provider for API eval: inline::meta-reference
> Enter provider for API telemetry: inline::meta-reference

 > (Optional) Enter a short description for your Llama Stack:

You can now edit ~/.llama/distributions/llamastack-my-local-stack/my-local-stack-run.yaml and run `llama stack run ~/.llama/distributions/llamastack-my-local-stack/my-local-stack-run.yaml`
```
:::

:::{tab-item} Building from a template
- To build from alternative API providers, we provide distribution templates for users to get started building a distribution backed by different providers.

The following command will allow you to see the available templates and their corresponding providers.
```
llama stack build --list-templates
```

```
+------------------------------+----------------------------------------+-----------------------------------------------------------------------------+
| Template Name                | Providers                              | Description                                                                 |
+------------------------------+----------------------------------------+-----------------------------------------------------------------------------+
| tgi                          | {                                      | Use (an external) TGI server for running LLM inference                      |
|                              |   "inference": [                       |                                                                             |
|                              |     "remote::tgi"                      |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "memory": [                          |                                                                             |
|                              |     "inline::faiss",                   |                                                                             |
|                              |     "remote::chromadb",                |                                                                             |
|                              |     "remote::pgvector"                 |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "safety": [                          |                                                                             |
|                              |     "inline::llama-guard"              |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "agents": [                          |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "telemetry": [                       |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ]                                    |                                                                             |
|                              | }                                      |                                                                             |
+------------------------------+----------------------------------------+-----------------------------------------------------------------------------+
| remote-vllm                  | {                                      | Use (an external) vLLM server for running LLM inference                     |
|                              |   "inference": [                       |                                                                             |
|                              |     "remote::vllm"                     |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "memory": [                          |                                                                             |
|                              |     "inline::faiss",                   |                                                                             |
|                              |     "remote::chromadb",                |                                                                             |
|                              |     "remote::pgvector"                 |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "safety": [                          |                                                                             |
|                              |     "inline::llama-guard"              |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "agents": [                          |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "telemetry": [                       |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ]                                    |                                                                             |
|                              | }                                      |                                                                             |
+------------------------------+----------------------------------------+-----------------------------------------------------------------------------+
| vllm-gpu                     | {                                      | Use a built-in vLLM engine for running LLM inference                        |
|                              |   "inference": [                       |                                                                             |
|                              |     "inline::vllm"                     |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "memory": [                          |                                                                             |
|                              |     "inline::faiss",                   |                                                                             |
|                              |     "remote::chromadb",                |                                                                             |
|                              |     "remote::pgvector"                 |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "safety": [                          |                                                                             |
|                              |     "inline::llama-guard"              |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "agents": [                          |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "telemetry": [                       |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ]                                    |                                                                             |
|                              | }                                      |                                                                             |
+------------------------------+----------------------------------------+-----------------------------------------------------------------------------+
| meta-reference-quantized-gpu | {                                      | Use Meta Reference with fp8, int4 quantization for running LLM inference    |
|                              |   "inference": [                       |                                                                             |
|                              |     "inline::meta-reference-quantized" |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "memory": [                          |                                                                             |
|                              |     "inline::faiss",                   |                                                                             |
|                              |     "remote::chromadb",                |                                                                             |
|                              |     "remote::pgvector"                 |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "safety": [                          |                                                                             |
|                              |     "inline::llama-guard"              |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "agents": [                          |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "telemetry": [                       |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ]                                    |                                                                             |
|                              | }                                      |                                                                             |
+------------------------------+----------------------------------------+-----------------------------------------------------------------------------+
| meta-reference-gpu           | {                                      | Use Meta Reference for running LLM inference                                |
|                              |   "inference": [                       |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "memory": [                          |                                                                             |
|                              |     "inline::faiss",                   |                                                                             |
|                              |     "remote::chromadb",                |                                                                             |
|                              |     "remote::pgvector"                 |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "safety": [                          |                                                                             |
|                              |     "inline::llama-guard"              |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "agents": [                          |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "telemetry": [                       |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ]                                    |                                                                             |
|                              | }                                      |                                                                             |
+------------------------------+----------------------------------------+-----------------------------------------------------------------------------+
| hf-serverless                | {                                      | Use (an external) Hugging Face Inference Endpoint for running LLM inference |
|                              |   "inference": [                       |                                                                             |
|                              |     "remote::hf::serverless"           |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "memory": [                          |                                                                             |
|                              |     "inline::faiss",                   |                                                                             |
|                              |     "remote::chromadb",                |                                                                             |
|                              |     "remote::pgvector"                 |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "safety": [                          |                                                                             |
|                              |     "inline::llama-guard"              |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "agents": [                          |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "telemetry": [                       |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ]                                    |                                                                             |
|                              | }                                      |                                                                             |
+------------------------------+----------------------------------------+-----------------------------------------------------------------------------+
| together                     | {                                      | Use Together.AI for running LLM inference                                   |
|                              |   "inference": [                       |                                                                             |
|                              |     "remote::together"                 |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "memory": [                          |                                                                             |
|                              |     "inline::faiss",                   |                                                                             |
|                              |     "remote::chromadb",                |                                                                             |
|                              |     "remote::pgvector"                 |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "safety": [                          |                                                                             |
|                              |     "inline::llama-guard"              |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "agents": [                          |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "telemetry": [                       |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ]                                    |                                                                             |
|                              | }                                      |                                                                             |
+------------------------------+----------------------------------------+-----------------------------------------------------------------------------+
| ollama                       | {                                      | Use (an external) Ollama server for running LLM inference                   |
|                              |   "inference": [                       |                                                                             |
|                              |     "remote::ollama"                   |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "memory": [                          |                                                                             |
|                              |     "inline::faiss",                   |                                                                             |
|                              |     "remote::chromadb",                |                                                                             |
|                              |     "remote::pgvector"                 |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "safety": [                          |                                                                             |
|                              |     "inline::llama-guard"              |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "agents": [                          |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "telemetry": [                       |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ]                                    |                                                                             |
|                              | }                                      |                                                                             |
+------------------------------+----------------------------------------+-----------------------------------------------------------------------------+
| bedrock                      | {                                      | Use AWS Bedrock for running LLM inference and safety                        |
|                              |   "inference": [                       |                                                                             |
|                              |     "remote::bedrock"                  |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "memory": [                          |                                                                             |
|                              |     "inline::faiss",                   |                                                                             |
|                              |     "remote::chromadb",                |                                                                             |
|                              |     "remote::pgvector"                 |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "safety": [                          |                                                                             |
|                              |     "remote::bedrock"                  |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "agents": [                          |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "telemetry": [                       |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ]                                    |                                                                             |
|                              | }                                      |                                                                             |
+------------------------------+----------------------------------------+-----------------------------------------------------------------------------+
| hf-endpoint                  | {                                      | Use (an external) Hugging Face Inference Endpoint for running LLM inference |
|                              |   "inference": [                       |                                                                             |
|                              |     "remote::hf::endpoint"             |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "memory": [                          |                                                                             |
|                              |     "inline::faiss",                   |                                                                             |
|                              |     "remote::chromadb",                |                                                                             |
|                              |     "remote::pgvector"                 |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "safety": [                          |                                                                             |
|                              |     "inline::llama-guard"              |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "agents": [                          |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "telemetry": [                       |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ]                                    |                                                                             |
|                              | }                                      |                                                                             |
+------------------------------+----------------------------------------+-----------------------------------------------------------------------------+
| fireworks                    | {                                      | Use Fireworks.AI for running LLM inference                                  |
|                              |   "inference": [                       |                                                                             |
|                              |     "remote::fireworks"                |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "memory": [                          |                                                                             |
|                              |     "inline::faiss",                   |                                                                             |
|                              |     "remote::chromadb",                |                                                                             |
|                              |     "remote::pgvector"                 |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "safety": [                          |                                                                             |
|                              |     "inline::llama-guard"              |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "agents": [                          |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "telemetry": [                       |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ]                                    |                                                                             |
|                              | }                                      |                                                                             |
+------------------------------+----------------------------------------+-----------------------------------------------------------------------------+
| cerebras                     | {                                      | Use Cerebras for running LLM inference                                      |
|                              |   "inference": [                       |                                                                             |
|                              |     "remote::cerebras"                 |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "safety": [                          |                                                                             |
|                              |     "inline::llama-guard"              |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "memory": [                          |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "agents": [                          |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ],                                   |                                                                             |
|                              |   "telemetry": [                       |                                                                             |
|                              |     "inline::meta-reference"           |                                                                             |
|                              |   ]                                    |                                                                             |
|                              | }                                      |                                                                             |
+------------------------------+----------------------------------------+-----------------------------------------------------------------------------+
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
:::

:::{tab-item} Building from a pre-existing build config file
- In addition to templates, you may customize the build to your liking through editing config files and build from config files with the following command.

- The config file will be of contents like the ones in `llama_stack/templates/*build.yaml`.

```
$ cat llama_stack/templates/ollama/build.yaml

name: ollama
distribution_spec:
  description: Like local, but use ollama for running LLM inference
  providers:
    inference: remote::ollama
    memory: inline::faiss
    safety: inline::llama-guard
    agents: meta-reference
    telemetry: meta-reference
image_type: conda
```

```
llama stack build --config llama_stack/templates/ollama/build.yaml
```
:::

:::{tab-item} Building Docker
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
:::

::::


## Running your Stack server
Now, let's start the Llama Stack Distribution Server. You will need the YAML configuration file which was written out at the end by the `llama stack build` step.

```
llama stack run ~/.llama/distributions/llamastack-my-local-stack/my-local-stack-run.yaml
```

```
$ llama stack run ~/.llama/distributions/llamastack-my-local-stack/my-local-stack-run.yaml

Serving API inspect
 GET /health
 GET /providers/list
 GET /routes/list
Serving API inference
 POST /inference/chat_completion
 POST /inference/completion
 POST /inference/embeddings
...
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

### Troubleshooting

If you encounter any issues, search through our [GitHub Issues](https://github.com/meta-llama/llama-stack/issues), or file an new issue.
