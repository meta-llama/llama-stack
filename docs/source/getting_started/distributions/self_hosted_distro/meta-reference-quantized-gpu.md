# Meta Reference Quantized Distribution

The `llamastack/distribution-meta-reference-quantized-gpu` distribution consists of the following provider configurations.


| **API**         	| **Inference**            	| **Agents**     	| **Memory**                                       	| **Safety**     	| **Telemetry**  	|
|-----------------	|------------------------  	|----------------	|--------------------------------------------------	|----------------	|----------------	|
| **Provider(s)** 	| meta-reference-quantized  | meta-reference 	| meta-reference, remote::pgvector, remote::chroma 	| meta-reference 	| meta-reference 	|

The only difference vs. the `meta-reference-gpu` distribution is that it has support for more efficient inference -- with fp8, int4 quantization, etc.

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
$ cd distributions/meta-reference-quantized-gpu && docker compose up
```

> [!NOTE]
> This assumes you have access to GPU to start a local server with access to your GPU.


> [!NOTE]
> `~/.llama` should be the path containing downloaded weights of Llama models.


This will download and start running a pre-built docker container. Alternatively, you may use the following commands:

```
docker run -it -p 5000:5000 -v ~/.llama:/root/.llama -v ./run.yaml:/root/my-run.yaml --gpus=all distribution-meta-reference-quantized-gpu --yaml_config /root/my-run.yaml
```

#### (Option 2) Start with Conda

1. Install the `llama` CLI. See [CLI Reference](https://llama-stack.readthedocs.io/en/latest/cli_reference/index.html)

2. Build the `meta-reference-quantized-gpu` distribution

```
$ llama stack build --template meta-reference-quantized-gpu --image-type conda
```

3. Start running distribution
```
$ cd distributions/meta-reference-quantized-gpu
$ llama stack run ./run.yaml
```