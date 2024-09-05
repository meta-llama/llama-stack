# Llama CLI Reference

The `llama` CLI tool helps you setup and use the Llama toolchain & agentic systems. It should be available on your path after installing the `llama-toolchain` package.

### Subcommands
1. `download`: `llama` cli tools supports downloading the model from Meta or HuggingFace.
2. `model`: Lists available models and their properties.
3. `stack`: Allows you to build and run a Llama Stack server. You can read more about this [here](/docs/cli_reference.md#step-3-building-configuring-and-running-llama-stack-servers).

### Sample Usage

```
llama --help
```
<pre style="font-family: monospace;">
usage: llama [-h] {download,model,stack} ...

Welcome to the Llama CLI

options:
  -h, --help            show this help message and exit

subcommands:
  {download,model,stack}
</pre>

## Step 1. Get the models

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

You can download from HuggingFace using these commands
Set your environment variable HF_TOKEN or pass in --hf-token to the command to validate your access.
You can find your token at [here](https://huggingface.co/settings/tokens)
```
llama download --source huggingface --model-id  Meta-Llama3.1-8B-Instruct --hf-token <HF_TOKEN>
```
```
llama download --source huggingface --model-id Meta-Llama3.1-70B-Instruct --hf-token <HF_TOKEN>
```

You can also download safety models from HF
```
llama download --source huggingface --model-id Llama-Guard-3-8B --ignore-patterns *original*
```
```
llama download --source huggingface --model-id Prompt-Guard-86M --ignore-patterns *original*
```

## Step 2: Understand the models
The `llama model` command helps you explore the model’s interface.

### 2.1 Subcommands
1. `download`: Download the model from different sources. (meta, huggingface)
2. `list`: Lists all the models available for download with hardware requirements to deploy the models.
3. `template`: <TODO: What is a template?>
4. `describe`: Describes all the properties of the model.

### 2.2 Sample Usage

`llama model <subcommand> <options>`

```
llama model --help
```
<pre style="font-family: monospace;">
usage: llama model [-h] {download,list,template,describe} ...

Work with llama models

options:
  -h, --help            show this help message and exit

model_subcommands:
  {download,list,template,describe}
</pre>

You can use the describe command to know more about a model:
```
llama model describe -m Meta-Llama3.1-8B-Instruct
```
### 2.3 Describe

<pre style="font-family: monospace;">
+-----------------------------+---------------------------------------+
| Model                       | Meta-                                 |
|                             | Llama3.1-8B-Instruct                  |
+-----------------------------+---------------------------------------+
| HuggingFace ID              | meta-llama/Meta-Llama-3.1-8B-Instruct |
+-----------------------------+---------------------------------------+
| Description                 | Llama 3.1 8b instruct model           |
+-----------------------------+---------------------------------------+
| Context Length              | 128K tokens                           |
+-----------------------------+---------------------------------------+
| Weights format              | bf16                                  |
+-----------------------------+---------------------------------------+
| Model params.json           | {                                     |
|                             |     "dim": 4096,                      |
|                             |     "n_layers": 32,                   |
|                             |     "n_heads": 32,                    |
|                             |     "n_kv_heads": 8,                  |
|                             |     "vocab_size": 128256,             |
|                             |     "ffn_dim_multiplier": 1.3,        |
|                             |     "multiple_of": 1024,              |
|                             |     "norm_eps": 1e-05,                |
|                             |     "rope_theta": 500000.0,           |
|                             |     "use_scaled_rope": true           |
|                             | }                                     |
+-----------------------------+---------------------------------------+
| Recommended sampling params | {                                     |
|                             |     "strategy": "top_p",              |
|                             |     "temperature": 1.0,               |
|                             |     "top_p": 0.9,                     |
|                             |     "top_k": 0                        |
|                             | }                                     |
+-----------------------------+---------------------------------------+
</pre>
### 2.4 Template
You can even run `llama model template` see all of the templates and their tokens:

```
llama model template
```

<pre style="font-family: monospace;">
+-----------+---------------------------------+
| Role      | Template Name                   |
+-----------+---------------------------------+
| user      | user-default                    |
| assistant | assistant-builtin-tool-call     |
| assistant | assistant-custom-tool-call      |
| assistant | assistant-default               |
| system    | system-builtin-and-custom-tools |
| system    | system-builtin-tools-only       |
| system    | system-custom-tools-only        |
| system    | system-default                  |
| tool      | tool-success                    |
| tool      | tool-failure                    |
+-----------+---------------------------------+
</pre>

And fetch an example by passing it to `--name`:
```
llama model template --name tool-success
```

<pre style="font-family: monospace;">
+----------+----------------------------------------------------------------+
| Name     | tool-success                                                   |
+----------+----------------------------------------------------------------+
| Template | <|start_header_id|>ipython<|end_header_id|>                    |
|          |                                                                |
|          | completed                                                      |
|          | [stdout]{"results":["something                                 |
|          | something"]}[/stdout]<|eot_id|>                                |
|          |                                                                |
+----------+----------------------------------------------------------------+
| Notes    | Note ipython header and [stdout]                               |
+----------+----------------------------------------------------------------+
</pre>

Or:
```
llama model template --name system-builtin-tools-only
```

<pre style="font-family: monospace;">
+----------+--------------------------------------------+
| Name     | system-builtin-tools-only                  |
+----------+--------------------------------------------+
| Template | <|start_header_id|>system<|end_header_id|> |
|          |                                            |
|          | Environment: ipython                       |
|          | Tools: brave_search, wolfram_alpha         |
|          |                                            |
|          | Cutting Knowledge Date: December 2023      |
|          | Today Date: 21 August 2024                 |
|          | <|eot_id|>                                 |
|          |                                            |
+----------+--------------------------------------------+
| Notes    |                                            |
+----------+--------------------------------------------+
</pre>

These commands can help understand the model interface and how prompts / messages are formatted for various scenarios.

**NOTE**: Outputs in terminal are color printed to show special tokens.


## Step 3: Listing, Building, and Configuring Llama Stack Distributions


### Step 3.1: List available distributions

Let’s start with listing available distributions:

```
llama stack list-distributions
```

<pre style="font-family: monospace;">
i+--------------------------------+---------------------------------------+----------------------------------------------------------------------+
| Distribution ID                | Providers                             | Description                                                          |
+--------------------------------+---------------------------------------+----------------------------------------------------------------------+
| local                          | {                                     | Use code from `llama_toolchain` itself to serve all llama stack APIs |
|                                |   "inference": "meta-reference",      |                                                                      |
|                                |   "memory": "meta-reference-faiss",   |                                                                      |
|                                |   "safety": "meta-reference",         |                                                                      |
|                                |   "agentic_system": "meta-reference"  |                                                                      |
|                                | }                                     |                                                                      |
+--------------------------------+---------------------------------------+----------------------------------------------------------------------+
| remote                         | {                                     | Point to remote services for all llama stack APIs                    |
|                                |   "inference": "remote",              |                                                                      |
|                                |   "safety": "remote",                 |                                                                      |
|                                |   "agentic_system": "remote",         |                                                                      |
|                                |   "memory": "remote"                  |                                                                      |
|                                | }                                     |                                                                      |
+--------------------------------+---------------------------------------+----------------------------------------------------------------------+
| local-ollama                   | {                                     | Like local, but use ollama for running LLM inference                 |
|                                |   "inference": "remote::ollama",      |                                                                      |
|                                |   "safety": "meta-reference",         |                                                                      |
|                                |   "agentic_system": "meta-reference", |                                                                      |
|                                |   "memory": "meta-reference-faiss"    |                                                                      |
|                                | }                                     |                                                                      |
+--------------------------------+---------------------------------------+----------------------------------------------------------------------+
| local-plus-fireworks-inference | {                                     | Use Fireworks.ai for running LLM inference                           |
|                                |   "inference": "remote::fireworks",   |                                                                      |
|                                |   "safety": "meta-reference",         |                                                                      |
|                                |   "agentic_system": "meta-reference", |                                                                      |
|                                |   "memory": "meta-reference-faiss"    |                                                                      |
|                                | }                                     |                                                                      |
+--------------------------------+---------------------------------------+----------------------------------------------------------------------+
| local-plus-together-inference  | {                                     | Use Together.ai for running LLM inference                            |
|                                |   "inference": "remote::together",    |                                                                      |
|                                |   "safety": "meta-reference",         |                                                                      |
|                                |   "agentic_system": "meta-reference", |                                                                      |
|                                |   "memory": "meta-reference-faiss"    |                                                                      |
|                                | }                                     |                                                                      |
+--------------------------------+---------------------------------------+----------------------------------------------------------------------+
</pre>

As you can see above, each “distribution” details the “providers” it is composed of. For example, `local` uses the “meta-reference” provider for inference while local-ollama relies on a different provider (Ollama) for inference. Similarly, you can use Fireworks or Together.AI for running inference as well.

### Step 3.2: Build a distribution

Let's imagine you are working with a 8B-Instruct model. The following command will build a package (in the form of a Conda environment) _and_ configure it. As part of the configuration, you will be asked for some inputs (model_id, max_seq_len, etc.) Since we are working with a 8B model, we will name our build `8b-instruct` to help us remember the config.

```
llama stack build local --name 8b-instruct
```

Once it runs successfully , you should see some outputs in the form:

```
$ llama stack build local --name 8b-instruct
....
....
Successfully installed cfgv-3.4.0 distlib-0.3.8 identify-2.6.0 libcst-1.4.0 llama_toolchain-0.0.2 moreorless-0.4.0 nodeenv-1.9.1 pre-commit-3.8.0 stdlibs-2024.5.15 toml-0.10.2 tomlkit-0.13.0 trailrunner-1.4.0 ufmt-2.7.0 usort-1.0.8 virtualenv-20.26.3

Successfully setup conda environment. Configuring build...

...
...

YAML configuration has been written to ~/.llama/builds/local/conda/8b-instruct.yaml
```
### Step 3.3: Configure a distribution

You can re-configure this distribution by running:
```
llama stack configure local --name 8b-instruct
```

Here is an example run of how the CLI will guide you to fill the configuration
```
$ llama stack configure local --name 8b-instruct

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

## Step 4: Starting a Llama Stack Distribution and Testing it

### Step 4.1: Starting a distribution

Now let’s start Llama Stack Distribution Server.

You need the YAML configuration file which was written out at the end by the `llama stack build` step.

```
llama stack run local --name 8b-instruct --port 5000
```
You should see the Stack server start and print the APIs that it is supporting,

```
$ llama stack run local --name 8b-instruct --port 5000

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

### Step 4.2: Test the distribution

Lets test with a client.
```
cd /path/to/llama-stack
conda activate <env>  # any environment containing the llama-toolchain pip package will work

python -m llama_toolchain.inference.client localhost 5000
```

This will run the chat completion client and query the distribution’s /inference/chat_completion API.

Here is an example output:
<pre style="font-family: monospace;">
Initializing client for http://localhost:5000
User>hello world, troll me in two-paragraphs about 42

Assistant> You think you're so smart, don't you? You think you can just waltz in here and ask about 42, like it's some kind of trivial matter. Well, let me tell you, 42 is not just a number, it's a way of life. It's the answer to the ultimate question of life, the universe, and everything, according to Douglas Adams' magnum opus, "The Hitchhiker's Guide to the Galaxy". But do you know what's even more interesting about 42? It's that it's not actually the answer to anything, it's just a number that some guy made up to sound profound.

You know what's even more hilarious? People like you who think they can just Google "42" and suddenly become experts on the subject. Newsflash: you're not a supercomputer, you're just a human being with a fragile ego and a penchant for thinking you're smarter than you actually are. 42 is just a number, a meaningless collection of digits that holds no significance whatsoever. So go ahead, keep thinking you're so clever, but deep down, you're just a pawn in the grand game of life, and 42 is just a silly little number that's been used to make you feel like you're part of something bigger than yourself. Ha!
</pre>

Similarly you can test safety (if you configured llama-guard and/or prompt-guard shields) by:

```
python -m llama_toolchain.safety.client localhost 5000
```
