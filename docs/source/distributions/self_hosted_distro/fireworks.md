---
orphan: true
---
# Fireworks Distribution

```{toctree}
:maxdepth: 2
:hidden:

self
```

The `llamastack/distribution-fireworks` distribution consists of the following provider configurations.

| API | Provider(s) |
|-----|-------------|
| agents | `inline::meta-reference` |
| datasetio | `remote::huggingface`, `inline::localfs` |
| eval | `inline::meta-reference` |
| inference | `remote::fireworks` |
| memory | `inline::faiss`, `remote::chromadb`, `remote::pgvector` |
| safety | `inline::llama-guard` |
| scoring | `inline::basic`, `inline::llm-as-judge`, `inline::braintrust` |
| telemetry | `inline::meta-reference` |
| tool_runtime | `remote::brave-search`, `remote::tavily-search`, `inline::code-interpreter`, `inline::memory-runtime` |


### Environment Variables

The following environment variables can be configured:

- `LLAMASTACK_PORT`: Port for the Llama Stack distribution server (default: `5001`)
- `FIREWORKS_API_KEY`: Fireworks.AI API Key (default: ``)

### Models

The following models are available by default:

- `meta-llama/Llama-3.1-8B-Instruct (fireworks/llama-v3p1-8b-instruct)`
- `meta-llama/Llama-3.1-70B-Instruct (fireworks/llama-v3p1-70b-instruct)`
- `meta-llama/Llama-3.1-405B-Instruct-FP8 (fireworks/llama-v3p1-405b-instruct)`
- `meta-llama/Llama-3.2-1B-Instruct (fireworks/llama-v3p2-1b-instruct)`
- `meta-llama/Llama-3.2-3B-Instruct (fireworks/llama-v3p2-3b-instruct)`
- `meta-llama/Llama-3.2-11B-Vision-Instruct (fireworks/llama-v3p2-11b-vision-instruct)`
- `meta-llama/Llama-3.2-90B-Vision-Instruct (fireworks/llama-v3p2-90b-vision-instruct)`
- `meta-llama/Llama-3.3-70B-Instruct (fireworks/llama-v3p3-70b-instruct)`
- `meta-llama/Llama-Guard-3-8B (fireworks/llama-guard-3-8b)`
- `meta-llama/Llama-Guard-3-11B-Vision (fireworks/llama-guard-3-11b-vision)`


### Prerequisite: API Keys

Make sure you have access to a Fireworks API Key. You can get one by visiting [fireworks.ai](https://fireworks.ai/).


## Running Llama Stack with Fireworks

You can do this via Conda (build code) or Docker which has a pre-built image.

### Via Docker

This method allows you to get started quickly without having to build the distribution code.

```bash
LLAMA_STACK_PORT=5001
docker run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  llamastack/distribution-fireworks \
  --port $LLAMA_STACK_PORT \
  --env FIREWORKS_API_KEY=$FIREWORKS_API_KEY
```

### Via Conda

```bash
llama stack build --template fireworks --image-type conda
llama stack run ./run.yaml \
  --port $LLAMA_STACK_PORT \
  --env FIREWORKS_API_KEY=$FIREWORKS_API_KEY
```
