# Configuring a Stack

The Llama Stack runtime configuration is specified as a YAML file. Here is a simplied version of an example configuration file for the Ollama distribution:

```{dropdown} Sample Configuration File
:closed:

```yaml
version: 2
conda_env: ollama
apis:
- agents
- inference
- memory
- safety
- telemetry
providers:
  inference:
  - provider_id: ollama
    provider_type: remote::ollama
    config:
      url: ${env.OLLAMA_URL:http://localhost:11434}
  memory:
  - provider_id: faiss
    provider_type: inline::faiss
    config:
      kvstore:
        type: sqlite
        namespace: null
        db_path: ${env.SQLITE_STORE_DIR:~/.llama/distributions/ollama}/faiss_store.db
  safety:
  - provider_id: llama-guard
    provider_type: inline::llama-guard
    config: {}
  agents:
  - provider_id: meta-reference
    provider_type: inline::meta-reference
    config:
      persistence_store:
        type: sqlite
        namespace: null
        db_path: ${env.SQLITE_STORE_DIR:~/.llama/distributions/ollama}/agents_store.db
  telemetry:
  - provider_id: meta-reference
    provider_type: inline::meta-reference
    config: {}
metadata_store:
  namespace: null
  type: sqlite
  db_path: ${env.SQLITE_STORE_DIR:~/.llama/distributions/ollama}/registry.db
models:
- metadata: {}
  model_id: ${env.INFERENCE_MODEL}
  provider_id: ollama
  provider_model_id: null
shields: []
```

Let's break this down into the different sections. It starts by specifying the set of APIs that the stack server will serve:
```yaml
apis:
- agents
- inference
- memory
- safety
- telemetry
```

Next up is the most critical section -- the set of providers that the stack will use to serve the above APIs. Let's take the `inference` API as an example:
```yaml
providers:
  inference:
  - provider_id: ollama
    provider_type: remote::ollama
    config:
      url: ${env.OLLAMA_URL:http://localhost:11434}
```
A _provider instance_ is identified with an (identifier, type, configuration) tuple. The identifier is a string you can choose freely. You may instantiate any number of provider instances of the same type. The configuration dictionary is provider-specific. Notice that configuration can reference environment variables (with default values), which are expanded at runtime. When you run a stack server (via docker or via `llama stack run`), you can specify `--env OLLAMA_URL=http://my-server:11434` to override the default value.

Finally, let's look at the `models` section:
```yaml
models:
- metadata: {}
  model_id: ${env.INFERENCE_MODEL}
  provider_id: ollama
  provider_model_id: null
```
A Model is an instance of a "Resource" (see [Concepts](../concepts)) and is associated with a specific inference provider (in this case, the provider with identifier `ollama`). This is an instance of a "pre-registered" model. While we always encourage the clients to always register models before using them, some Stack servers may come up a list of "already known and available" models.

What's with the `provider_model_id` field? This is an identifier for the model inside the provider's model catalog. Contrast it with `model_id` which is the identifier for the same model for Llama Stack's purposes. For example, you may want to name "llama3.2:vision-11b" as "image_captioning_model" when you use it in your Stack interactions. When omitted, the server will set `provider_model_id` to be the same as `model_id`.
