# Llama Stack Integration Tests
You can run llama stack integration tests on either a Llama Stack Library or a Llama Stack endpoint.

To test on a Llama Stack library with certain configuration, run
```bash
LLAMA_STACK_CONFIG=./llama_stack/templates/cerebras/run.yaml
pytest -s -v tests/client-sdk/inference/
```
or just the template name
```bash
LLAMA_STACK_CONFIG=together
pytest -s -v tests/client-sdk/inference/
```

To test on a Llama Stack endpoint, run
```bash
LLAMA_STACK_BASE_URL=http//localhost:8089
pytest -s -v tests/client-sdk/inference
```

## Report Generation

To generate a report, run with `--report` option
```bash
LLAMA_STACK_CONFIG=together pytest -s -v report.md tests/client-sdk/ --report
```

## Common options
Depending on the API, there are custom options enabled
- For tests in `inference/` and `agents/, we support `--inference-model` (to be used in text inference tests) and `--vision-inference-model` (only used in image inference tests) overrides
- For tests in `vector_io/`, we support `--embedding-model` override
- For tests in `safety/`, we support `--safety-shield` override
- The param can be `--report` or `--report <path>`
If path is not provided, we do a best effort to infer based on the config / template name. For url endpoints, path is required.
