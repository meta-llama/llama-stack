# Configuring a Stack

The Llama Stack runtime configuration is specified as a YAML file. Here is a simplied version of an example configuration file for the Ollama distribution:

```{dropdown} Sample Configuration File

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

Let's break this down into the different sections. The first section specifies the set of APIs that the stack server will serve:
```yaml
apis:
- agents
- inference
- memory
- safety
- telemetry
```

## Providers
Next up is the most critical part: the set of providers that the stack will use to serve the above APIs. Consider the `inference` API:
```yaml
providers:
  inference:
  - provider_id: ollama
    provider_type: remote::ollama
    config:
      url: ${env.OLLAMA_URL:http://localhost:11434}
```
A few things to note:
- A _provider instance_ is identified with an (identifier, type, configuration) tuple. The identifier is a string you can choose freely.
- You can instantiate any number of provider instances of the same type.
- The configuration dictionary is provider-specific. Notice that configuration can reference environment variables (with default values), which are expanded at runtime. When you run a stack server (via docker or via `llama stack run`), you can specify `--env OLLAMA_URL=http://my-server:11434` to override the default value.

## Resources
Finally, let's look at the `models` section:
```yaml
models:
- metadata: {}
  model_id: ${env.INFERENCE_MODEL}
  provider_id: ollama
  provider_model_id: null
```
A Model is an instance of a "Resource" (see [Concepts](../concepts/index)) and is associated with a specific inference provider (in this case, the provider with identifier `ollama`). This is an instance of a "pre-registered" model. While we always encourage the clients to always register models before using them, some Stack servers may come up a list of "already known and available" models.

What's with the `provider_model_id` field? This is an identifier for the model inside the provider's model catalog. Contrast it with `model_id` which is the identifier for the same model for Llama Stack's purposes. For example, you may want to name "llama3.2:vision-11b" as "image_captioning_model" when you use it in your Stack interactions. When omitted, the server will set `provider_model_id` to be the same as `model_id`.

## Extending to handle Safety

Configuring Safety can be a little involved so it is instructive to go through an example.

The Safety API works with the associated Resource called a `Shield`. Providers can support various kinds of Shields. Good examples include the [Llama Guard](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/) system-safety models, or [Bedrock Guardrails](https://aws.amazon.com/bedrock/guardrails/).

To configure a Bedrock Shield, you would need to add:
- A Safety API provider instance with type `remote::bedrock`
- A Shield resource served by this provider.

```yaml
...
providers:
  safety:
  - provider_id: bedrock
    provider_type: remote::bedrock
    config:
      aws_access_key_id: ${env.AWS_ACCESS_KEY_ID}
      aws_secret_access_key: ${env.AWS_SECRET_ACCESS_KEY}
...
shields:
- provider_id: bedrock
  params:
    guardrailVersion: ${env.GUARDRAIL_VERSION}
  provider_shield_id: ${env.GUARDRAIL_ID}
...
```

The situation is more involved if the Shield needs _Inference_ of an associated model. This is the case with Llama Guard. In that case, you would need to add:
- A Safety API provider instance with type `inline::llama-guard`
- An Inference API provider instance for serving the model.
- A Model resource associated with this provider.
- A Shield resource served by the Safety provider.

The yaml configuration for this setup, assuming you were using vLLM as your inference server, would look like:
```yaml
...
providers:
  safety:
  - provider_id: llama-guard
    provider_type: inline::llama-guard
    config: {}
  inference:
  # this vLLM server serves the "normal" inference model (e.g., llama3.2:3b)
  - provider_id: vllm-0
    provider_type: remote::vllm
    config:
      url: ${env.VLLM_URL:http://localhost:8000}
  # this vLLM server serves the llama-guard model (e.g., llama-guard:3b)
  - provider_id: vllm-1
    provider_type: remote::vllm
    config:
      url: ${env.SAFETY_VLLM_URL:http://localhost:8001}
...
models:
- metadata: {}
  model_id: ${env.INFERENCE_MODEL}
  provider_id: vllm-0
  provider_model_id: null
- metadata: {}
  model_id: ${env.SAFETY_MODEL}
  provider_id: vllm-1
  provider_model_id: null
shields:
- provider_id: llama-guard
  shield_id: ${env.SAFETY_MODEL}   # Llama Guard shields are identified by the corresponding LlamaGuard model
  provider_shield_id: null
...
```
