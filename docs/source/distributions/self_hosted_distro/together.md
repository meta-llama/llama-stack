---
orphan: true
---
# Together Distribution

```{toctree}
:maxdepth: 2
:hidden:

self
```

The `llamastack/distribution-together` distribution consists of the following provider configurations.

| API | Provider(s) |
|-----|-------------|
| agents | `inline::meta-reference` |
| datasetio | `remote::huggingface`, `inline::localfs` |
| eval | `inline::meta-reference` |
| inference | `remote::together` |
| memory | `inline::faiss`, `remote::chromadb`, `remote::pgvector` |
| safety | `inline::llama-guard` |
| scoring | `inline::basic`, `inline::llm-as-judge`, `inline::braintrust` |
| telemetry | `inline::meta-reference` |
| tool_runtime | `remote::brave-search`, `remote::tavily-search`, `inline::code-interpreter`, `inline::memory-runtime` |


### Environment Variables

The following environment variables can be configured:

- `LLAMA_STACK_PORT`: Port for the Llama Stack distribution server (default: `5001`)
- `TOGETHER_API_KEY`: Together.AI API Key (default: ``)

### Models

The following models are available by default:

- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.1-70B-Instruct`
- `meta-llama/Llama-3.1-405B-Instruct-FP8`
- `meta-llama/Llama-3.2-3B-Instruct`
- `meta-llama/Llama-3.2-11B-Vision-Instruct`
- `meta-llama/Llama-3.2-90B-Vision-Instruct`
- `meta-llama/Llama-3.3-70B-Instruct`
- `meta-llama/Llama-Guard-3-8B`
- `meta-llama/Llama-Guard-3-11B-Vision`


### Prerequisite: API Keys

Make sure you have access to a Together API Key. You can get one by visiting [together.xyz](https://together.xyz/).


## Running Llama Stack with Together

You can do this via Conda (build code) or Docker which has a pre-built image.

### Via Docker

This method allows you to get started quickly without having to build the distribution code.

```bash
LLAMA_STACK_PORT=5001
docker run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  llamastack/distribution-together \
  --port $LLAMA_STACK_PORT \
  --env TOGETHER_API_KEY=$TOGETHER_API_KEY
```

### Via Conda

```bash
llama stack build --template together --image-type conda
llama stack run ./run.yaml \
  --port $LLAMA_STACK_PORT \
  --env TOGETHER_API_KEY=$TOGETHER_API_KEY
```
