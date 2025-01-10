---
orphan: true
---
# Meta Reference Quantized Distribution

```{toctree}
:maxdepth: 2
:hidden:

self
```

The `llamastack/distribution-meta-reference-quantized-gpu` distribution consists of the following provider configurations:

| API | Provider(s) |
|-----|-------------|
| agents | `inline::meta-reference` |
| datasetio | `remote::huggingface`, `inline::localfs` |
| eval | `inline::meta-reference` |
| inference | `inline::meta-reference-quantized` |
| memory | `inline::faiss`, `remote::chromadb`, `remote::pgvector` |
| safety | `inline::llama-guard` |
| scoring | `inline::basic`, `inline::llm-as-judge`, `inline::braintrust` |
| telemetry | `inline::meta-reference` |
| tool_runtime | `remote::brave-search`, `remote::tavily-search`, `inline::code-interpreter`, `inline::memory-runtime` |


The only difference vs. the `meta-reference-gpu` distribution is that it has support for more efficient inference -- with fp8, int4 quantization, etc.

Note that you need access to nvidia GPUs to run this distribution. This distribution is not compatible with CPU-only machines or machines with AMD GPUs.

### Environment Variables

The following environment variables can be configured:

- `LLAMA_STACK_PORT`: Port for the Llama Stack distribution server (default: `5001`)
- `INFERENCE_MODEL`: Inference model loaded into the Meta Reference server (default: `meta-llama/Llama-3.2-3B-Instruct`)
- `INFERENCE_CHECKPOINT_DIR`: Directory containing the Meta Reference model checkpoint (default: `null`)


## Prerequisite: Downloading Models

Please make sure you have llama model checkpoints downloaded in `~/.llama` before proceeding. See [installation guide](https://llama-stack.readthedocs.io/en/latest/references/llama_cli_reference/download_models.html) here to download the models. Run `llama model list` to see the available models to download, and `llama model download` to download the checkpoints.

```
$ ls ~/.llama/checkpoints
Llama3.1-8B           Llama3.2-11B-Vision-Instruct  Llama3.2-1B-Instruct  Llama3.2-90B-Vision-Instruct  Llama-Guard-3-8B
Llama3.1-8B-Instruct  Llama3.2-1B                   Llama3.2-3B-Instruct  Llama-Guard-3-1B              Prompt-Guard-86M
```

## Running the Distribution

You can do this via Conda (build code) or Docker which has a pre-built image.

### Via Docker

This method allows you to get started quickly without having to build the distribution code.

```bash
LLAMA_STACK_PORT=5001
docker run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  llamastack/distribution-meta-reference-quantized-gpu \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct
```

If you are using Llama Stack Safety / Shield APIs, use:

```bash
docker run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  llamastack/distribution-meta-reference-quantized-gpu \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct \
  --env SAFETY_MODEL=meta-llama/Llama-Guard-3-1B
```

### Via Conda

Make sure you have done `pip install llama-stack` and have the Llama Stack CLI available.

```bash
llama stack build --template meta-reference-quantized-gpu --image-type conda
llama stack run distributions/meta-reference-quantized-gpu/run.yaml \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct
```

If you are using Llama Stack Safety / Shield APIs, use:

```bash
llama stack run distributions/meta-reference-quantized-gpu/run-with-safety.yaml \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct \
  --env SAFETY_MODEL=meta-llama/Llama-Guard-3-1B
```
