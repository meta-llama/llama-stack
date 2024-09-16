# Getting Started

The `llama` CLI tool helps you setup and use the Llama toolchain & agentic systems. It should be available on your path after installing the `llama-toolchain` package.

This guides allows you to quickly get started with building and running a Llama Stack server in < 5 minutes!

## Quick Cheatsheet
- Quick 3 line command to build and start a LlamaStack server using our Meta Reference implementation for all API endpoints with `conda` as build type.

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
In the following steps, imagine we'll be working with a `Meta-Llama3.1-8B-Instruct` model. We will name our build `8b-instruct` to help us remember the config. We will start build our distribution (in the form of a Conda environment, or Docker image). In this step, we will specify:
- `name`: the name for our distribution (e.g. `8b-instruct`)
- `image_type`: our build image type (`conda | docker`)
- `distribution_spec`: our distribution specs for specifying API providers
  - `description`: a short description of the configurations for the distribution
  - `providers`: specifies the underlying implementation for serving each API endpoint
  - `image_type`: `conda` | `docker` to specify whether to build the distribution in the form of Docker image or Conda environment.

#### Build a local distribution with conda
The following command and specifications allows you to get started with building.
```
llama stack build <path/to/config>
```
- You will be required to pass in a file path to the build.config file (e.g. `./llama_toolchain/configs/distributions/conda/local-conda-example-build.yaml`). We provide some example build config files for configuring different types of distributions in the `./llama_toolchain/configs/distributions/` folder.

The file will be of the contents
```
$ cat ./llama_toolchain/configs/distributions/conda/local-conda-example-build.yaml

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

You may run the `llama stack build` command to generate your distribution with `--name` to override the name for your distribution.
```
$ llama stack build ~/.llama/distributions/conda/8b-instruct-build.yaml --name 8b-instruct
...
...
Build spec configuration saved at ~/.llama/distributions/conda/8b-instruct-build.yaml
```

After this step is complete, a file named `8b-instruct-build.yaml` will be generated and saved at `~/.llama/distributions/conda/8b-instruct-build.yaml`.


#### How to build distribution with different API providers using configs
To specify a different API provider, we can change the `distribution_spec` in our `<name>-build.yaml` config. For example, the following build spec allows you to build a distribution using TGI as the inference API provider.

```
$ cat ./llama_toolchain/configs/distributions/conda/local-tgi-conda-example-build.yaml

name: local-tgi-conda-example
distribution_spec:
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
llama stack configure [<path/to/name.build.yaml> | <docker-image-name>]
```
- For `conda` environments: <path/to/name.build.yaml> would be the generated build spec saved from Step 1.
- For `docker` images downloaded from Dockerhub, you could also use <docker-image-name> as the argument.
   - Run `docker images` to check list of available images on your machine.

```
$ llama stack configure ~/.llama/distributions/conda/8b-instruct-build.yaml

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
llama stack run ~/.llama/builds/conda/8b-instruct-run.yaml
```

You should see the Llama Stack server start and print the APIs that it is supporting

```
$ llama stack run ~/.llama/builds/local/conda/8b-instruct.yaml

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
Serving POST /safety/run_shields
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
> Configuration is in `~/.llama/builds/local/conda/8b-instruct.yaml`. Feel free to increase `max_seq_len`.

> [!IMPORTANT]
> The "local" distribution inference server currently only supports CUDA. It will not work on Apple Silicon machines.

> [!TIP]
> You might need to use the flag `--disable-ipv6` to  Disable IPv6 support

This server is running a Llama model locally.

## Step 4. Test with Client
Once the server is setup, we can test it with a client to see the example outputs.
```
cd /path/to/llama-stack
conda activate <env>  # any environment containing the llama-toolchain pip package will work

python -m llama_toolchain.inference.client localhost 5000
```

This will run the chat completion client and query the distribution’s /inference/chat_completion API.

Here is an example output:
```
Initializing client for http://localhost:5000
User>hello world, troll me in two-paragraphs about 42

Assistant> You think you're so smart, don't you? You think you can just waltz in here and ask about 42, like it's some kind of trivial matter. Well, let me tell you, 42 is not just a number, it's a way of life. It's the answer to the ultimate question of life, the universe, and everything, according to Douglas Adams' magnum opus, "The Hitchhiker's Guide to the Galaxy". But do you know what's even more interesting about 42? It's that it's not actually the answer to anything, it's just a number that some guy made up to sound profound.

You know what's even more hilarious? People like you who think they can just Google "42" and suddenly become experts on the subject. Newsflash: you're not a supercomputer, you're just a human being with a fragile ego and a penchant for thinking you're smarter than you actually are. 42 is just a number, a meaningless collection of digits that holds no significance whatsoever. So go ahead, keep thinking you're so clever, but deep down, you're just a pawn in the grand game of life, and 42 is just a silly little number that's been used to make you feel like you're part of something bigger than yourself. Ha!
```

Similarly you can test safety (if you configured llama-guard and/or prompt-guard shields) by:

```
python -m llama_toolchain.safety.client localhost 5000
```

You can find more example scripts with client SDKs to talk with the Llama Stack server in our [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/sdk_examples) repo.
