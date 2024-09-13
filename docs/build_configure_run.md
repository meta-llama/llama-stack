# Get Started

The `llama` CLI tool helps you setup and use the Llama toolchain & agentic systems. It should be available on your path after installing the `llama-toolchain` package.

This guides allows you to quickly get started with building and running a Llama Stack server in < 5 minutes!

### TL;DR
Let's imagine you are working with a 8B-Instruct model. We will name our build `8b-instruct` to help us remember the config.

**llama stack build**
```
llama stack build

Enter value for name (required): 8b-instruct
Enter value for distribution (default: local) (required):
Enter value for api_providers (optional):
Enter value for image_type (default: conda) (required):

...
Build spec configuration saved at ~/.llama/distributions/local/docker/8b-instruct-build.yaml
```

**llama stack configure**
```
$ llama stack configure ~/.llama/distributions/local/docker/8b-instruct-build.yaml
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

YAML configuration has been written to ~/.llama/builds/local/docker/8b-instruct-build.yaml
```

**llama stack run**
```
llama stack run ~/.llama/builds/local/docker/8b-instruct-build.yaml
...
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
INFO:     Started server process [3403915]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://[::]:5000 (Press CTRL+C to quit)
```

### Step 0. Prerequisites
You first need to have models downloaded locally.

To download any model you need the **Model Descriptor**.
This can be obtained by running the command
```
llama model list
```

You should see a table like this:

<pre style="font-family: monospace;">
+---------------------------------------+---------------------------------------------+----------------+----------------------------+
| Model Descriptor                      | HuggingFace Repo                            | Context Length | Hardware Requirements      |
+---------------------------------------+---------------------------------------------+----------------+----------------------------+
| Meta-Llama3.1-8B                      | meta-llama/Meta-Llama-3.1-8B                | 128K           | 1 GPU, each >= 20GB VRAM   |
+---------------------------------------+---------------------------------------------+----------------+----------------------------+
| Meta-Llama3.1-70B                     | meta-llama/Meta-Llama-3.1-70B               | 128K           | 8 GPUs, each >= 20GB VRAM  |
+---------------------------------------+---------------------------------------------+----------------+----------------------------+
| Meta-Llama3.1-405B:bf16-mp8           |                                             | 128K           | 8 GPUs, each >= 120GB VRAM |
+---------------------------------------+---------------------------------------------+----------------+----------------------------+
| Meta-Llama3.1-405B                    | meta-llama/Meta-Llama-3.1-405B-FP8          | 128K           | 8 GPUs, each >= 70GB VRAM  |
+---------------------------------------+---------------------------------------------+----------------+----------------------------+
| Meta-Llama3.1-405B:bf16-mp16          | meta-llama/Meta-Llama-3.1-405B              | 128K           | 16 GPUs, each >= 70GB VRAM |
+---------------------------------------+---------------------------------------------+----------------+----------------------------+
| Meta-Llama3.1-8B-Instruct             | meta-llama/Meta-Llama-3.1-8B-Instruct       | 128K           | 1 GPU, each >= 20GB VRAM   |
+---------------------------------------+---------------------------------------------+----------------+----------------------------+
| Meta-Llama3.1-70B-Instruct            | meta-llama/Meta-Llama-3.1-70B-Instruct      | 128K           | 8 GPUs, each >= 20GB VRAM  |
+---------------------------------------+---------------------------------------------+----------------+----------------------------+
| Meta-Llama3.1-405B-Instruct:bf16-mp8  |                                             | 128K           | 8 GPUs, each >= 120GB VRAM |
+---------------------------------------+---------------------------------------------+----------------+----------------------------+
| Meta-Llama3.1-405B-Instruct           | meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 | 128K           | 8 GPUs, each >= 70GB VRAM  |
+---------------------------------------+---------------------------------------------+----------------+----------------------------+
| Meta-Llama3.1-405B-Instruct:bf16-mp16 | meta-llama/Meta-Llama-3.1-405B-Instruct     | 128K           | 16 GPUs, each >= 70GB VRAM |
+---------------------------------------+---------------------------------------------+----------------+----------------------------+
| Llama-Guard-3-8B                      | meta-llama/Llama-Guard-3-8B                 | 128K           | 1 GPU, each >= 20GB VRAM   |
+---------------------------------------+---------------------------------------------+----------------+----------------------------+
| Llama-Guard-3-8B:int8-mp1             | meta-llama/Llama-Guard-3-8B-INT8            | 128K           | 1 GPU, each >= 10GB VRAM   |
+---------------------------------------+---------------------------------------------+----------------+----------------------------+
| Prompt-Guard-86M                      | meta-llama/Prompt-Guard-86M                 | 128K           | 1 GPU, each >= 1GB VRAM    |
+---------------------------------------+---------------------------------------------+----------------+----------------------------+
</pre>

To download models, you can use the llama download command.

Here is an example download command to get the 8B/70B Instruct model. You will need META_URL which can be obtained from [here](https://llama.meta.com/docs/getting_the_models/meta/)
```
llama download --source meta --model-id Meta-Llama3.1-8B-Instruct --meta-url <META_URL>
```
```
llama download --source meta --model-id Meta-Llama3.1-70B-Instruct --meta-url <META_URL>
```

### Step 1. Build

##### Build conda
Let's imagine you are working with a 8B-Instruct model. The following command will build a package (in the form of a Conda environment).  Since we are working with a 8B model, we will name our build `8b-instruct` to help us remember the config.


```
llama stack build
```

```
$ llama stack build

Enter value for name (required): 8b-instruct
Enter value for distribution (default: local) (required):
Enter value for api_providers (optional):
Enter value for image_type (default: conda) (required):

....
....
Successfully installed cfgv-3.4.0 distlib-0.3.8 identify-2.6.0 libcst-1.4.0 llama_toolchain-0.0.2 moreorless-0.4.0 nodeenv-1.9.1 pre-commit-3.8.0 stdlibs-2024.5.15 toml-0.10.2 tomlkit-0.13.0 trailrunner-1.4.0 ufmt-2.7.0 usort-1.0.8 virtualenv-20.26.3
...
...
Build spec configuration saved at /home/xiyan/.llama/distributions/local/conda/8b-instruct-build.yaml
```

##### Build docker
The following command will build a package (in the form of a Docker container). Since we are working with a 8B model, we will name our build `8b-instruct` to help us remember the config. We will specify the `image_type` as `docker` to build Docker container.

```
$ llama stack build

Enter value for name (required): 8b-instruct
Enter value for distribution (default: local) (required):
Enter value for api_providers (optional):
Enter value for image_type (default: conda) (required): docker

...
...
COMMIT llamastack-d
--> a319efac9f0a
Successfully tagged localhost/llamastack-d:latest
a319efac9f0a488d18662b90efdb863df6c1a2c9cffaea6e247e4abd90b1bfc2
+ set +x
Succesfully setup Podman image. Configuring build...You can run it with: podman run -p 8000:8000 llamastack-d
Build spec configuration saved at /home/xiyan/.llama/distributions/local/docker/d-build.yaml
```

##### Re-build from config
You can re-build package based on build config
```
$ cat ~/.llama/distributions/local/conda/8b-instruct-build.yaml
name: 8b-instruct
distribution: local
api_providers: null
image_type: conda

$ llama stack build --config ~/.llama/distributions/local/conda/8b-instruct-build.yaml

Successfully setup conda environment. Configuring build...

...
...
Build spec configuration saved at ~/.llama/distributions/local/conda/8b-instruct-build.yaml
```

### Step 2. Configure

Next, you will need to configure the distribution to specify run settings for running the server. As part of the configuration, you will be asked for some inputs (model_id, max_seq_len, etc.) You should configure this distribution by running:
```
llama stack configure ~/.llama/builds/local/conda/8b-instruct-build.yaml
```

Here is an example run of how the CLI will guide you to fill the configuration

```
$ llama stack configure ~/.llama/builds/local/conda/8b-instruct-build.yaml

Configuring API: inference (meta-reference)
Enter value for model (required): Meta-Llama3.1-8B-Instruct
Enter value for quantization (optional):
Enter value for torch_seed (optional):
Enter value for max_seq_len (required): 4096
Enter value for max_batch_size (default: 1): 1
Configuring API: safety (meta-reference)
Do you want to configure llama_guard_shield? (y/n): y
Entering sub-configuration for llama_guard_shield:
Enter value for model (required): Llama-Guard-3-8B
Enter value for excluded_categories (required): []
Enter value for disable_input_check (default: False):
Enter value for disable_output_check (default: False):
Do you want to configure prompt_guard_shield? (y/n): y
Entering sub-configuration for prompt_guard_shield:
Enter value for model (required): Prompt-Guard-86M
...
...
YAML configuration has been written to ~/.llama/builds/local/conda/8b-instruct.yaml
```

As you can see, we did basic configuration above and configured:
- inference to run on model `Meta-Llama3.1-8B-Instruct` (obtained from `llama model list`)
- Llama Guard safety shield with model `Llama-Guard-3-8B`
- Prompt Guard safety shield with model `Prompt-Guard-86M`

For how these configurations are stored as yaml, checkout the file printed at the end of the configuration.

Note that all configurations as well as models are stored in `~/.llama`

### Step 3. Run

Now let’s start Llama Stack Distribution Server.

You need the YAML configuration file which was written out at the end by the `llama stack configure` step.

```
llama stack run ~/.llama/builds/local/conda/8b-instruct.yaml
```
You should see the Stack server start and print the APIs that it is supporting,

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

This server is running a Llama model locally.
