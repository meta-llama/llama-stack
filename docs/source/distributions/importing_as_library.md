# Using Llama Stack as a Library

If you are planning to use an external service for Inference (even Ollama or TGI counts as external), it is often easier to use Llama Stack as a library. This avoids the overhead of setting up a server. For [example](https://github.com/meta-llama/llama-stack-client-python/blob/main/src/llama_stack_client/lib/direct/test.py):

```python
from llama_stack_client.lib.direct.direct import LlamaStackDirectClient

client = await LlamaStackDirectClient.from_template('ollama')
await client.initialize()
```

This will parse your config and set up any inline implementations and remote clients needed for your implementation.

Then, you can access the APIs like `models` and `inference` on the client and call their methods directly:

```python
response = await client.models.list()
print(response)
```

```python
response = await client.inference.chat_completion(
    messages=[UserMessage(content="What is the capital of France?", role="user")],
    model="Llama3.1-8B-Instruct",
    stream=False,
)
print("\nChat completion response:")
print(response)
```

If you've created a [custom distribution](https://llama-stack.readthedocs.io/en/latest/distributions/building_distro.html), you can also use the run.yaml configuration file directly:

```python
client = await LlamaStackDirectClient.from_config(config_path)
await client.initialize()
```
