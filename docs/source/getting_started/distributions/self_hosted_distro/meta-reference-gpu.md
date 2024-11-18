# Meta Reference Distribution

The `llamastack/distribution-meta-reference-gpu` distribution consists of the following provider configurations:

| API | Provider(s) |
|-----|-------------|
| agents | `inline::meta-reference` |
| inference | `inline::meta-reference` |
| memory | `inline::faiss`, `remote::chromadb`, `remote::pgvector` |
| safety | `inline::llama-guard` |
| telemetry | `inline::meta-reference` |


Note that you need access to nvidia GPUs to run this distribution. This distribution is not compatible with CPU-only machines or machines with AMD GPUs.



## Prerequisite: Downloading Models

Please make sure you have llama model checkpoints downloaded in `~/.llama` before proceeding. See [installation guide](https://llama-stack.readthedocs.io/en/latest/cli_reference/download_models.html) here to download the models. Run `llama model list` to see the available models to download, and `llama model download` to download the checkpoints.

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
  -v ./run.yaml:/root/my-run.yaml \
  llamastack/distribution-meta-reference-gpu \
  /root/my-run.yaml \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct
```

If you are using Llama Stack Safety / Shield APIs, use:

```bash
docker run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ./run-with-safety.yaml:/root/my-run.yaml \
  llamastack/distribution-meta-reference-gpu \
  /root/my-run.yaml \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct \
  --env SAFETY_MODEL=meta-llama/Llama-Guard-3-1B
```

### Via Conda

Make sure you have done `pip install llama-stack` and have the Llama Stack CLI available.

```bash
llama stack build --template meta-reference-gpu --image-type conda
llama stack run ./run.yaml \
  --port 5001 \
  --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct
```

If you are using Llama Stack Safety / Shield APIs, use:

```bash
llama stack run ./run-with-safety.yaml \
  --port 5001 \
  --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct \
  --env SAFETY_MODEL=meta-llama/Llama-Guard-3-1B
```
