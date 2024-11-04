# Meta Reference Distribution

The `llamastack/distribution-meta-reference-gpu` distribution consists of the following provider configurations.


| **API**         	| **Inference** 	| **Agents**     	| **Memory**                                       	| **Safety**     	| **Telemetry**  	|
|-----------------	|---------------	|----------------	|--------------------------------------------------	|----------------	|----------------	|
| **Provider(s)** 	| meta-reference  	| meta-reference 	| meta-reference, remote::pgvector, remote::chroma 	| meta-reference 	| meta-reference 	|


### Step 0. Prerequisite - Downloading Models
Please make sure you have llama model checkpoints downloaded in `~/.llama` before proceeding. See [installation guide](https://llama-stack.readthedocs.io/en/latest/cli_reference/download_models.html) here to download the models.

```
$ ls ~/.llama/checkpoints
Llama3.1-8B           Llama3.2-11B-Vision-Instruct  Llama3.2-1B-Instruct  Llama3.2-90B-Vision-Instruct  Llama-Guard-3-8B
Llama3.1-8B-Instruct  Llama3.2-1B                   Llama3.2-3B-Instruct  Llama-Guard-3-1B              Prompt-Guard-86M
```

### Step 1. Start the Distribution

#### (Option 1) Start with Docker
```
$ cd distributions/meta-reference-gpu && docker compose up
```

> [!NOTE]
> This assumes you have access to GPU to start a local server with access to your GPU.


> [!NOTE]
> `~/.llama` should be the path containing downloaded weights of Llama models.


This will download and start running a pre-built docker container. Alternatively, you may use the following commands:

```
docker run -it -p 5000:5000 -v ~/.llama:/root/.llama -v ./run.yaml:/root/my-run.yaml --gpus=all distribution-meta-reference-gpu --yaml_config /root/my-run.yaml
```

#### (Option 2) Start with Conda

1. Install the `llama` CLI. See [CLI Reference](https://llama-stack.readthedocs.io/en/latest/cli_reference/index.html)

2. Build the `meta-reference-gpu` distribution

```
$ llama stack build --template meta-reference-gpu --image-type conda
```

3. Start running distribution
```
$ cd distributions/meta-reference-gpu
$ llama stack run ./run.yaml
```

### (Optional) Serving a new model
You may change the `config.model` in `run.yaml` to update the model currently being served by the distribution. Make sure you have the model checkpoint downloaded in your `~/.llama`.
```
inference:
  - provider_id: meta0
    provider_type: meta-reference
    config:
      model: Llama3.2-11B-Vision-Instruct
      quantization: null
      torch_seed: null
      max_seq_len: 4096
      max_batch_size: 1
```

Run `llama model list` to see the available models to download, and `llama model download` to download the checkpoints.
