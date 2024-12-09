# NVIDIA Distribution

The `llamastack/distribution-nvidia` distribution consists of the following provider configurations.

| API | Provider(s) |
|-----|-------------|
| agents | `inline::meta-reference` |
| inference | `remote::nvidia` |
| memory | `inline::faiss`, `remote::chromadb`, `remote::pgvector` |
| safety | `inline::llama-guard` |
| telemetry | `inline::meta-reference` |


### Environment Variables

The following environment variables can be configured:

- `LLAMASTACK_PORT`: Port for the Llama Stack distribution server (default: `5001`)
- `NVIDIA_API_KEY`: NVIDIA API Key (default: ``)

### Models

The following models are available by default:

- `${env.INFERENCE_MODEL} (None)`


### Prerequisite: API Keys

Make sure you have access to a NVIDIA API Key. You can get one by visiting [https://build.nvidia.com/](https://build.nvidia.com/).


## Running Llama Stack with NVIDIA

You can do this via Conda (build code) or Docker which has a pre-built image.

### Via Docker

This method allows you to get started quickly without having to build the distribution code.

```bash
LLAMA_STACK_PORT=5001
docker run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ./run.yaml:/root/my-run.yaml \
  llamastack/distribution-nvidia \
  --yaml-config /root/my-run.yaml \
  --port $LLAMA_STACK_PORT \
  --env NVIDIA_API_KEY=$NVIDIA_API_KEY
```

### Via Conda

```bash
llama stack build --template nvidia --image-type conda
llama stack run ./run.yaml \
  --port 5001 \
  --env NVIDIA_API_KEY=$NVIDIA_API_KEY
```