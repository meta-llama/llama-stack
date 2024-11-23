# Importing Llama Stack as a Python Library

Llama Stack is typically utilized in a client-server configuration. To get started quickly, you can import Llama Stack as a library and call the APIs directly without needing to set up a server. For [example](https://github.com/meta-llama/llama-stack-client-python/blob/main/src/llama_stack_client/lib/direct/test.py):

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

If you've created a [custom distribution](https://llama-stack.readthedocs.io/en/latest/distributions/building_distro.html), you can also import it with the `from_config` constructor:

```python
import yaml

with open(config_path, "r") as f:
    config_dict = yaml.safe_load(f)

run_config = parse_and_maybe_upgrade_config(config_dict)

client = await LlamaStackDirectClient.from_config(run_config)
```
