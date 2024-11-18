# Fireworks Distribution

The `llamastack/distribution-fireworks` distribution consists of the following provider configurations.

| API | Provider(s) |
|-----|-------------|
| agents | `inline::meta-reference` |
| inference | `remote::fireworks` |
| memory | `inline::faiss`, `remote::chromadb`, `remote::pgvector` |
| safety | `inline::llama-guard` |
| telemetry | `inline::meta-reference` |


### Environment Variables

The following environment variables can be configured:

- `LLAMASTACK_PORT`: Port for the Llama Stack distribution server (default: `5001`)
- `FIREWORKS_API_KEY`: Fireworks.AI API Key (default: ``)

### Models

The following models are available by default:

- `fireworks/llama-v3p1-8b-instruct`
- `fireworks/llama-v3p1-70b-instruct`
- `fireworks/llama-v3p1-405b-instruct`
- `fireworks/llama-v3p2-1b-instruct`
- `fireworks/llama-v3p2-3b-instruct`
- `fireworks/llama-v3p2-11b-vision-instruct`
- `fireworks/llama-v3p2-90b-vision-instruct`
- `fireworks/llama-guard-3-8b`
- `fireworks/llama-guard-3-11b-vision`


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
  -v ./run.yaml:/root/my-run.yaml \
  llamastack/distribution-fireworks \
  /root/my-run.yaml \
  --port $LLAMA_STACK_PORT \
  --env FIREWORKS_API_KEY=$FIREWORKS_API_KEY
```

### Via Conda

```bash
llama stack build --template fireworks --image-type conda
llama stack run ./run.yaml \
  --port 5001 \
  --env FIREWORKS_API_KEY=$FIREWORKS_API_KEY
```
