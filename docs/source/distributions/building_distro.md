# Build your own Distribution


This guide will walk you through the steps to get started with building a Llama Stack distribution from scratch with your choice of API providers.


### Llama Stack Build

In order to build your own distribution, we recommend you clone the `llama-stack` repository.


```
git clone git@github.com:meta-llama/llama-stack.git
cd llama-stack
pip install -e .
```
Use the CLI to build your distribution.
The main points to consider are:
1. **Image Type** - Do you want a Conda / venv environment or a Container (eg. Docker)
2. **Template** - Do you want to use a template to build your distribution? or start from scratch ?
3. **Config** - Do you want to use a pre-existing config file to build your distribution?

```
llama stack build -h

usage: llama stack build [-h] [--config CONFIG] [--template TEMPLATE] [--list-templates | --no-list-templates] [--image-type {conda,container,venv}] [--image-name IMAGE_NAME]

Build a Llama stack container

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to a config file to use for the build. You can find example configs in llama_stack/distribution/**/build.yaml.
                        If this argument is not provided, you will be prompted to enter information interactively
  --template TEMPLATE   Name of the example template config to use for build. You may use `llama stack build --list-templates` to check out the available templates
  --list-templates, --no-list-templates
                        Show the available templates for building a Llama Stack distribution (default: False)
  --image-type {conda,container,venv}
                        Image Type to use for the build. This can be either conda or container or venv. If not specified, will use the image type from the template config.
  --image-name IMAGE_NAME
                        [for image-type=conda] Name of the conda environment to use for the build. If
                        not specified, currently active Conda environment will be used. If no Conda
                        environment is active, you must specify a name.
```

After this step is complete, a file named `<name>-build.yaml` and template file `<name>-run.yaml` will be generated and saved at the output file path specified at the end of the command.

::::{tab-set}
:::{tab-item} Building from a template
To build from alternative API providers, we provide distribution templates for users to get started building a distribution backed by different providers.

The following command will allow you to see the available templates and their corresponding providers.
```
llama stack build --list-templates
```

```
------------------------------+-----------------------------------------------------------------------------+
| Template Name                | Description                                                                 |
+------------------------------+-----------------------------------------------------------------------------+
| hf-serverless                | Use (an external) Hugging Face Inference Endpoint for running LLM inference |
+------------------------------+-----------------------------------------------------------------------------+
| together                     | Use Together.AI for running LLM inference                                   |
+------------------------------+-----------------------------------------------------------------------------+
| vllm-gpu                     | Use a built-in vLLM engine for running LLM inference                        |
+------------------------------+-----------------------------------------------------------------------------+
| experimental-post-training   | Experimental template for post training                                     |
+------------------------------+-----------------------------------------------------------------------------+
| remote-vllm                  | Use (an external) vLLM server for running LLM inference                     |
+------------------------------+-----------------------------------------------------------------------------+
| fireworks                    | Use Fireworks.AI for running LLM inference                                  |
+------------------------------+-----------------------------------------------------------------------------+
| tgi                          | Use (an external) TGI server for running LLM inference                      |
+------------------------------+-----------------------------------------------------------------------------+
| bedrock                      | Use AWS Bedrock for running LLM inference and safety                        |
+------------------------------+-----------------------------------------------------------------------------+
| meta-reference-gpu           | Use Meta Reference for running LLM inference                                |
+------------------------------+-----------------------------------------------------------------------------+
| nvidia                       | Use NVIDIA NIM for running LLM inference                                    |
+------------------------------+-----------------------------------------------------------------------------+
| meta-reference-quantized-gpu | Use Meta Reference with fp8, int4 quantization for running LLM inference    |
+------------------------------+-----------------------------------------------------------------------------+
| cerebras                     | Use Cerebras for running LLM inference                                      |
+------------------------------+-----------------------------------------------------------------------------+
| ollama                       | Use (an external) Ollama server for running LLM inference                   |
+------------------------------+-----------------------------------------------------------------------------+
| hf-endpoint                  | Use (an external) Hugging Face Inference Endpoint for running LLM inference |
+------------------------------+-----------------------------------------------------------------------------+
```

You may then pick a template to build your distribution with providers fitted to your liking.

For example, to build a distribution with TGI as the inference provider, you can run:
```
$ llama stack build --template tgi
...
You can now edit ~/.llama/distributions/llamastack-tgi/tgi-run.yaml and run `llama stack run ~/.llama/distributions/llamastack-tgi/tgi-run.yaml`
```
:::
:::{tab-item} Building from Scratch

If the provided templates do not fit your use case, you could start off with running `llama stack build` which will allow you to a interactively enter wizard where you will be prompted to enter build configurations.

It would be best to start with a template and understand the structure of the config file and the various concepts ( APIS, providers, resources, etc.) before starting from scratch.
```
llama stack build

> Enter a name for your Llama Stack (e.g. my-local-stack): my-stack
> Enter the image type you want your Llama Stack to be built as (container or conda): conda

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
    agents: inline::meta-reference
    telemetry: inline::meta-reference
image_type: conda
```

```
llama stack build --config llama_stack/templates/ollama/build.yaml
```
:::

:::{tab-item} Building Container
> [!TIP]
> Podman is supported as an alternative to Docker. Set `CONTAINER_BINARY` to `podman` in your environment to use Podman.

To build a container image, you may start off from a template and use the `--image-type container` flag to specify `container` as the build image type.

```
llama stack build --template ollama --image-type container
```

```
$ llama stack build --template ollama --image-type container
...
Containerfile created successfully in /tmp/tmp.viA3a3Rdsg/ContainerfileFROM python:3.10-slim
...

You can now edit ~/meta-llama/llama-stack/tmp/configs/ollama-run.yaml and run `llama stack run ~/meta-llama/llama-stack/tmp/configs/ollama-run.yaml`
```

After this step is successful, you should be able to find the built container image and test it with `llama stack run <path/to/run.yaml>`.
:::

::::


### Running your Stack server
Now, let's start the Llama Stack Distribution Server. You will need the YAML configuration file which was written out at the end by the `llama stack build` step.

```
# Start using template name
llama stack run tgi

# Start using config file
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

Listening on ['::', '0.0.0.0']:8321
INFO:     Started server process [2935911]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://['::', '0.0.0.0']:8321 (Press CTRL+C to quit)
INFO:     2401:db00:35c:2d2b:face:0:c9:0:54678 - "GET /models/list HTTP/1.1" 200 OK
```

### Troubleshooting

If you encounter any issues, ask questions in our discord or search through our [GitHub Issues](https://github.com/meta-llama/llama-stack/issues), or file an new issue.
