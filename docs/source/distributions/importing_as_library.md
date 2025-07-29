# Using Llama Stack as a Library

## Setup Llama Stack without a Server
If you are planning to use an external service for Inference (even Ollama or TGI counts as external), it is often easier to use Llama Stack as a library.
This avoids the overhead of setting up a server.
```bash
# setup
uv pip install llama-stack
llama stack build --template starter --image-type venv
```

```python
from llama_stack.distribution.library_client import LlamaStackAsLibraryClient

client = LlamaStackAsLibraryClient(
    "starter",
    # provider_data is optional, but if you need to pass in any provider specific data, you can do so here.
    provider_data={"tavily_search_api_key": os.environ["TAVILY_SEARCH_API_KEY"]},
)
client.initialize()
```

This will parse your config and set up any inline implementations and remote clients needed for your implementation.

Then, you can access the APIs like `models` and `inference` on the client and call their methods directly:

```python
response = client.models.list()
```

If you've created a [custom distribution](https://llama-stack.readthedocs.io/en/latest/distributions/building_distro.html), you can also use the run.yaml configuration file directly:

```python
client = LlamaStackAsLibraryClient(config_path)
client.initialize()
```
