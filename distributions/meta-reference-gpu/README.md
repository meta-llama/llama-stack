# Meta Reference Distribution

The `llamastack/distribution-meta-reference-gpu` distribution consists of the following provider configurations.


| **API**         	| **Inference** 	| **Agents**     	| **Memory**                                       	| **Safety**     	| **Telemetry**  	|
|-----------------	|---------------	|----------------	|--------------------------------------------------	|----------------	|----------------	|
| **Provider(s)** 	| meta-reference  	| meta-reference 	| meta-reference, remote::pgvector, remote::chroma 	| meta-reference 	| meta-reference 	|


### Start the Distribution (Single Node GPU)

```
$ cd distributions/meta-reference-gpu
$ ls
build.yaml  compose.yaml  README.md  run.yaml
$ docker compose up
```

> [!NOTE]
> This assumes you have access to GPU to start a local server with access to your GPU.


> [!NOTE]
> `~/.llama` should be the path containing downloaded weights of Llama models.


This will download and start running a pre-built docker container. Alternatively, you may use the following commands:

```
docker run -it -p 5000:5000 -v ~/.llama:/root/.llama -v ./run.yaml:/root/my-run.yaml --gpus=all distribution-meta-reference-gpu --yaml_config /root/my-run.yaml
```

### Alternative (Build and start distribution locally via conda)
- You may checkout the [Getting Started](../../docs/getting_started.md) for more details on building locally via conda and starting up a meta-reference distribution.

### Start Distribution With pgvector/chromadb Memory Provider
##### pgvector
1. Start running the pgvector server:

```
docker run --network host --name mypostgres -it -p 5432:5432 -e POSTGRES_PASSWORD=mysecretpassword -e POSTGRES_USER=postgres -e POSTGRES_DB=postgres pgvector/pgvector:pg16
```

2. Edit the `run.yaml` file to point to the pgvector server.
```
memory:
  - provider_id: pgvector
    provider_type: remote::pgvector
    config:
      host: 127.0.0.1
      port: 5432
      db: postgres
      user: postgres
      password: mysecretpassword
```

> [!NOTE]
> If you get a `RuntimeError: Vector extension is not installed.`. You will need to run `CREATE EXTENSION IF NOT EXISTS vector;` to include the vector extension. E.g.

```
docker exec -it mypostgres ./bin/psql -U postgres
postgres=# CREATE EXTENSION IF NOT EXISTS vector;
postgres=# SELECT extname from pg_extension;
 extname
```

3. Run `docker compose up` with the updated `run.yaml` file.

##### chromadb
1. Start running chromadb server
```
docker run -it --network host --name chromadb -p 6000:6000 -v ./chroma_vdb:/chroma/chroma -e IS_PERSISTENT=TRUE chromadb/chroma:latest
```

2. Edit the `run.yaml` file to point to the chromadb server.
```
memory:
  - provider_id: remote::chromadb
    provider_type: remote::chromadb
    config:
      host: localhost
      port: 6000
```

3. Run `docker compose up` with the updated `run.yaml` file.

### Serving a new model
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
