# Getting Started

The `llama` CLI tool helps you setup and use the Llama toolchain & agentic systems. It should be available on your path after installing the `llama-toolchain` package.

This guides allows you to quickly get started with building and running a Llama Stack server in < 5 minutes!

In the following steps, we'll be working with a 8B-Instruct model. Since we are working with a 8B model, we will name our build `8b-instruct` to help us remember the config.

## Quick Cheatsheet
- Quick 3 line command to build and start a LlamaStack server using our Meta Reference implementation for all API endpoints.

**`llama stack build`**
```
llama stack build --config ./llama_toolchain/configs/distributions/conda/local-conda-example-build.yaml --name my-local-llama-stack
...
...
Build spec configuration saved at ~/.llama/distributions/conda/my-local-llama-stack-build.yaml
```

**`llama stack configure`**
```
llama stack configure ~/.llama/distributions/conda/my-local-llama-stack-build.yaml

Configuring API: inference (meta-reference)
Enter value for model (default: Meta-Llama3.1-8B-Instruct) (required):
Enter value for quantization (optional):
Enter value for torch_seed (optional):
Enter value for max_seq_len (required): 4096
Enter value for max_batch_size (default: 1) (required):

Configuring API: memory (meta-reference-faiss)

Configuring API: safety (meta-reference)
Do you want to configure llama_guard_shield? (y/n): n
Do you want to configure prompt_guard_shield? (y/n): n

Configuring API: agentic_system (meta-reference)
Enter value for brave_search_api_key (optional):
Enter value for bing_search_api_key (optional):
Enter value for wolfram_api_key (optional):

Configuring API: telemetry (console)

YAML configuration has been written to ~/.llama/builds/conda/my-local-llama-stack-run.yaml
```

**`llama stack run`**
```
llama stack run ~/.llama/builds/conda/my-local-llama-stack-run.yaml

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
Serving POST /safety/run_shields
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
We will start build our distribution (in the form of a Conda environment, or Docker image). In this step, we will specify:
- `name`: the name for our distribution (e.g. `8b-instruct`)
- `image_type`: our build image type (`conda | docker`)
- `distribution_spec`: our distribution specs for specifying API providers
  - `distribution_type`: an unique name to identify our distribution. The available distributions can be found in [llama_toolchain/configs/distributions/distribution_registry](llama_toolchain/configs/distributions/distribution_registry/) folder in the form of YAML files. You can run `llama stack list-distributions` to see the available distributions.
  - `description`: a short description of the configurations for the distribution
  - `providers`: specifies the underlying implementation for serving each API endpoint
  - `image_type`: `conda` | `docker` to specify whether to build the distribution in the form of Docker image or Conda environment.

#### Build a local distribution with conda
The following command and specifications allows you to get started with building.
```
llama stack build
```

You will be prompted to enter config specifications.
```
$ llama stack build

Enter value for name (required): 8b-instruct

Entering sub-configuration for distribution_spec:
Enter value for distribution_type (default: local) (required):
Enter value for description (default: Use code from `llama_toolchain` itself to serve all llama stack APIs) (required):
Enter value for docker_image (optional):
Enter value for providers (default: {'inference': 'meta-reference', 'memory': 'meta-reference-faiss', 'safety': 'meta-reference', 'agentic_system': 'meta-reference', 'telemetry': 'console'}) (required):
Enter value for image_type (default: conda) (required):

Conda environment 'llamastack-8b-instruct' exists. Checking Python version...

Build spec configuration saved at ~/.llama/distributions/conda/8b-instruct-build.yaml
```

After this step is complete, a file named `8b-instruct-build.yaml` will be generated and saved at `~/.llama/distributions/conda/8b-instruct-build.yaml`.

The file will be of the contents
```
$ cat ~/.llama/distributions/conda/8b-instruct-build.yaml

name: 8b-instruct
distribution_spec:
  distribution_type: local
  description: Use code from `llama_toolchain` itself to serve all llama stack APIs
  docker_image: null
  providers:
    inference: meta-reference
    memory: meta-reference-faiss
    safety: meta-reference
    agentic_system: meta-reference
    telemetry: console
image_type: conda
```

You may edit the `8b-instruct-build.yaml` file and re-run the `llama stack build` command to re-build and update the distribution.
```
llama stack build --config ~/.llama/distributions/conda/8b-instruct-build.yaml
```

#### How to build distribution with different API providers using configs
To specify a different API provider, we can change the `distribution_spec` in our `<name>-build.yaml` config. For example, the following build spec allows you to build a distribution using TGI as the inference API provider.

```
$ cat ./llama_toolchain/configs/distributions/conda/local-tgi-conda-example-build.yaml

name: local-tgi-conda-example
distribution_spec:
  distribution_type: local-plus-tgi-inference
  description: Use TGI (local or with Hugging Face Inference Endpoints for running LLM inference. When using HF Inference Endpoints, you must provide the name of the endpoint).
  docker_image: null
  providers:
    inference: remote::tgi
    memory: meta-reference-faiss
    safety: meta-reference
    agentic_system: meta-reference
    telemetry: console
image_type: conda
```

The following command allows you to build a distribution with TGI as the inference API provider, with the name `tgi`.
```
llama stack build --config ./llama_toolchain/configs/distributions/conda/local-tgi-conda-example-build.yaml --name tgi
```

We provide some example build configs to help you get started with building with different API providers.

#### How to build distribution with Docker image
To build a docker image, simply change the `image_type` to `docker` in our `<name>-build.yaml` file, and run `llama stack build --config <name>-build.yaml`.

```
$ cat ./llama_toolchain/configs/distributions/docker/local-docker-example-build.yaml

name: local-docker-example
distribution_spec:
  distribution_type: local
  description: Use code from `llama_toolchain` itself to serve all llama stack APIs
  docker_image: null
  providers:
    inference: meta-reference
    memory: meta-reference-faiss
    safety: meta-reference
    agentic_system: meta-reference
    telemetry: console
image_type: docker
```

The following command allows you to build a Docker image with the name `docker-local`
```
llama stack build --config ./llama_toolchain/configs/distributions/docker/local-docker-example-build.yaml --name docker-local

Dockerfile created successfully in /tmp/tmp.I0ifS2c46A/DockerfileFROM python:3.10-slim
WORKDIR /app
...
...
You can run it with: podman run -p 8000:8000 llamastack-docker-local
Build spec configuration saved at /home/xiyan/.llama/distributions/docker/docker-local-build.yaml
```

## Step 2. Configure
After our distribution is built (either in form of docker or conda environment), we will run the following command to
```
llama stack configure <path/to/name.build.yaml>
```

```
$ llama stack configure
```

> TODO: For Docker, specify docker image instead of build config.


## Step 3. Run
