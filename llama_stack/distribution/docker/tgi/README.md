# TGI GPU Distribution

The scripts in these folders help you spin up a Llama Stack distribution with TGI Inference provider.

> [!NOTE]
> This assumes you have access to GPU to start a TGI server with access to your GPU. Please see TGI CPU Distribution if you wish connect to a hosted TGI endpoint.

### Getting Started

```
$ cd llama_stack/distribution/docker/tgi
$ ls
compose.yaml  tgi-run.yaml
$ docker compose up
```

The script will first start up TGI server, then start up Llama Stack distribution server hooking up to the remote TGI provider for inference. You should be able to see the following outputs --
```
[text-generation-inference] | 2024-10-15T18:56:33.810397Z  INFO text_generation_router::server: router/src/server.rs:1813: Using config Some(Llama)
[text-generation-inference] | 2024-10-15T18:56:33.810448Z  WARN text_generation_router::server: router/src/server.rs:1960: Invalid hostname, defaulting to 0.0.0.0
[text-generation-inference] | 2024-10-15T18:56:33.864143Z  INFO text_generation_router::server: router/src/server.rs:2353: Connected
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://[::]:5000 (Press CTRL+C to quit)
```

To kill the server
```
docker compose down
```

### (Alternative) Docker Run

If you wish to separately spin up a TGI server, and connect with Llama Stack, you may use the following commands.

##### Start TGI server.
```
docker run --rm -it -v $HOME/.cache/huggingface:/data -p 5009:5009 --gpus all ghcr.io/huggingface/text-generation-inference:latest --dtype bfloat16 --usage-stats on --sharded false --model-id meta-llama/Llama-3.1-8B-Instruct --port 5009
```
- Please check the [TGI Getting Started Guide](https://github.com/huggingface/text-generation-inference?tab=readme-ov-file#get-started) for more details.

##### Start Llama Stack server pointing to TGI server

```
docker run --network host -it -p 5000:5000 -v ~/.llama:/root/.llama -v ./tgi-run.yaml:/root/llamastack-run-tgi.yaml --gpus=all llamastack-local-cpu --yaml_config /root/llamastack-run-tgi.yaml
```

Make sure in you `tgi-run.yaml` file, you inference provider is pointing to the correct TGI endpoint. E.g.
```
inference:
  - provider_id: tgi0
    provider_type: remote::tgi
    config:
      url: http://127.0.0.1:5009
```
