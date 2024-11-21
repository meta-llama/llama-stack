# Bedrock Distribution

```{toctree}
:maxdepth: 2
:hidden:

self
```

The `llamastack/distribution-bedrock` distribution consists of the following provider configurations:

| API | Provider(s) |
|-----|-------------|
| agents | `inline::meta-reference` |
| inference | `remote::bedrock` |
| memory | `inline::faiss`, `remote::chromadb`, `remote::pgvector` |
| safety | `remote::bedrock` |
| telemetry | `inline::meta-reference` |



### Environment Variables

The following environment variables can be configured:

- `LLAMASTACK_PORT`: Port for the Llama Stack distribution server (default: `5001`)



### Prerequisite: API Keys

Make sure you have access to a AWS Bedrock API Key. You can get one by visiting [AWS Bedrock](https://aws.amazon.com/bedrock/).


## Running Llama Stack with AWS Bedrock

You can do this via Conda (build code) or Docker which has a pre-built image.

### Via Docker

This method allows you to get started quickly without having to build the distribution code.

```bash
LLAMA_STACK_PORT=5001
docker run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  llamastack/distribution-bedrock \
  --port $LLAMA_STACK_PORT \
  --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  --env AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN
```

### Via Conda

```bash
llama stack build --template bedrock --image-type conda
llama stack run ./run.yaml \
  --port $LLAMA_STACK_PORT \
  --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  --env AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN
```
