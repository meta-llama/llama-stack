# Memory API Providers

This guide gives you references to switch between different memory API providers.

##### pgvector
1. Start running the pgvector server:

```
$ docker run --network host --name mypostgres -it -p 5432:5432 -e POSTGRES_PASSWORD=mysecretpassword -e POSTGRES_USER=postgres -e POSTGRES_DB=postgres pgvector/pgvector:pg16
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
