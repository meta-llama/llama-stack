# Llama Stack Integration Tests

We use `pytest` for parameterizing and running tests. You can see all options with:
```bash
cd tests/integration

# this will show a long list of options, look for "Custom options:"
pytest --help
```

Here are the most important options:
- `--stack-config`: specify the stack config to use. You have four ways to point to a stack:
  - **`server:<config>`** - automatically start a server with the given config (e.g., `server:fireworks`). This provides one-step testing by auto-starting the server if the port is available, or reusing an existing server if already running.
  - **`server:<config>:<port>`** - same as above but with a custom port (e.g., `server:together:8322`)
  - a URL which points to a Llama Stack distribution server
  - a template (e.g., `starter`) or a path to a `run.yaml` file
  - a comma-separated list of api=provider pairs, e.g. `inference=fireworks,safety=llama-guard,agents=meta-reference`. This is most useful for testing a single API surface.
- `--env`: set environment variables, e.g. --env KEY=value. this is a utility option to set environment variables required by various providers.

Model parameters can be influenced by the following options:
- `--text-model`: comma-separated list of text models.
- `--vision-model`: comma-separated list of vision models.
- `--embedding-model`: comma-separated list of embedding models.
- `--safety-shield`: comma-separated list of safety shields.
- `--judge-model`: comma-separated list of judge models.
- `--embedding-dimension`: output dimensionality of the embedding model to use for testing. Default: 384

Each of these are comma-separated lists and can be used to generate multiple parameter combinations. Note that tests will be skipped
if no model is specified.

## Examples

### Testing against a Server

Run all text inference tests by auto-starting a server with the `fireworks` config:

```bash
pytest -s -v tests/integration/inference/test_text_inference.py \
   --stack-config=server:fireworks \
   --text-model=meta-llama/Llama-3.1-8B-Instruct
```

Run tests with auto-server startup on a custom port:

```bash
pytest -s -v tests/integration/inference/ \
   --stack-config=server:together:8322 \
   --text-model=meta-llama/Llama-3.1-8B-Instruct
```

Run multiple test suites with auto-server (eliminates manual server management):

```bash
# Auto-start server and run all integration tests
export FIREWORKS_API_KEY=<your_key>

pytest -s -v tests/integration/inference/ tests/integration/safety/ tests/integration/agents/ \
   --stack-config=server:fireworks \
   --text-model=meta-llama/Llama-3.1-8B-Instruct
```

### Testing with Library Client

Run all text inference tests with the `starter` distribution using the `together` provider:

```bash
ENABLE_TOGETHER=together pytest -s -v tests/integration/inference/test_text_inference.py \
   --stack-config=starter \
   --text-model=meta-llama/Llama-3.1-8B-Instruct
```

Run all text inference tests with the `starter` distribution using the `together` provider and `meta-llama/Llama-3.1-8B-Instruct`:

```bash
ENABLE_TOGETHER=together pytest -s -v tests/integration/inference/test_text_inference.py \
   --stack-config=starter \
   --text-model=meta-llama/Llama-3.1-8B-Instruct
```

Running all inference tests for a number of models using the `together` provider:

```bash
TEXT_MODELS=meta-llama/Llama-3.1-8B-Instruct,meta-llama/Llama-3.1-70B-Instruct
VISION_MODELS=meta-llama/Llama-3.2-11B-Vision-Instruct
EMBEDDING_MODELS=all-MiniLM-L6-v2
ENABLE_TOGETHER=together
export TOGETHER_API_KEY=<together_api_key>

pytest -s -v tests/integration/inference/ \
   --stack-config=together \
   --text-model=$TEXT_MODELS \
   --vision-model=$VISION_MODELS \
   --embedding-model=$EMBEDDING_MODELS
```

Same thing but instead of using the distribution, use an adhoc stack with just one provider (`fireworks` for inference):

```bash
export FIREWORKS_API_KEY=<fireworks_api_key>

pytest -s -v tests/integration/inference/ \
   --stack-config=inference=fireworks \
   --text-model=$TEXT_MODELS \
   --vision-model=$VISION_MODELS \
   --embedding-model=$EMBEDDING_MODELS
```

Running Vector IO tests for a number of embedding models:

```bash
EMBEDDING_MODELS=all-MiniLM-L6-v2

pytest -s -v tests/integration/vector_io/ \
   --stack-config=inference=sentence-transformers,vector_io=sqlite-vec \
   --embedding-model=$EMBEDDING_MODELS
```
