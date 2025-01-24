# Llama Stack Integration Tests
You can run llama stack integration tests on either a Llama Stack Library or a Llama Stack endpoint.

To test on a Llama Stack library with certain configuration, run
```bash
LLAMA_STACK_CONFIG=./llama_stack/templates/cerebras/run.yaml
pytest -s -v tests/client-sdk/inference/test_inference.py
```

To test on a Llama Stack endpoint, run
```bash
LLAMA_STACK_BASE_URL=http//localhost:8089
pytest -s -v tests/client-sdk/inference/test_inference.py
```


## Common options
Depending on the API, there are custom options enabled
- For tests in `inference/` and `agents/, we support `--inference-model` (to be used in text inference tests) and `--vision-inference-model` (only used in image inference tests) overrides
- For tests in `vector_io/`, we support `--embedding-model` override
- For tests in `safety/`, we support `--safety-shield` override
