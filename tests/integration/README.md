# Llama Stack Integration Tests

We use `pytest` for parameterizing and running tests. You can see all options with:
```bash
cd tests/integration

# this will show a long list of options, look for "Custom options:"
pytest --help
```

Here are the most important options:
- `--stack-config`: specify the stack config to use. You have three ways to point to a stack:
  - a URL which points to a Llama Stack distribution server
  - a template (e.g., `fireworks`, `together`) or a path to a run.yaml file
  - a comma-separated list of api=provider pairs, e.g. `inference=fireworks,safety=llama-guard,agents=meta-reference`. This is most useful for testing a single API surface.
- `--env`: set environment variables, e.g. --env KEY=value. this is a utility option to set environment variables required by various providers.

Model parameters can be influenced by the following options:
- `--text-model`: comma-separated list of text models.
- `--vision-model`: comma-separated list of vision models.
- `--embedding-model`: comma-separated list of embedding models.
- `--safety-shield`: comma-separated list of safety shields.
- `--judge-model`: comma-separated list of judge models.
- `--embedding-dimension`: output dimensionality of the embedding model to use for testing. Default: 384

Each of these are comma-separated lists and can be used to generate multiple parameter combinations.


Experimental, under development, options:
- `--record-responses`: record new API responses instead of using cached ones
- `--report`: path where the test report should be written, e.g. --report=/path/to/report.md


## Examples

Run all text inference tests with the `together` distribution:

```bash
pytest -s -v tests/api/inference/test_text_inference.py \
   --stack-config=together \
   --text-model=meta-llama/Llama-3.1-8B-Instruct
```

Run all text inference tests with the `together` distribution and `meta-llama/Llama-3.1-8B-Instruct`:

```bash
pytest -s -v tests/api/inference/test_text_inference.py \
   --stack-config=together \
   --text-model=meta-llama/Llama-3.1-8B-Instruct
```

Running all inference tests for a number of models:

```bash
TEXT_MODELS=meta-llama/Llama-3.1-8B-Instruct,meta-llama/Llama-3.1-70B-Instruct
VISION_MODELS=meta-llama/Llama-3.2-11B-Vision-Instruct
EMBEDDING_MODELS=all-MiniLM-L6-v2
export TOGETHER_API_KEY=<together_api_key>

pytest -s -v tests/api/inference/ \
   --stack-config=together \
   --text-model=$TEXT_MODELS \
   --vision-model=$VISION_MODELS \
   --embedding-model=$EMBEDDING_MODELS
```

Same thing but instead of using the distribution, use an adhoc stack with just one provider (`fireworks` for inference):

```bash
export FIREWORKS_API_KEY=<fireworks_api_key>

pytest -s -v tests/api/inference/ \
   --stack-config=inference=fireworks \
   --text-model=$TEXT_MODELS \
   --vision-model=$VISION_MODELS \
   --embedding-model=$EMBEDDING_MODELS
```

Running Vector IO tests for a number of embedding models:

```bash
EMBEDDING_MODELS=all-MiniLM-L6-v2

pytest -s -v tests/api/vector_io/ \
   --stack-config=inference=sentence-transformers,vector_io=sqlite-vec \
   --embedding-model=$EMBEDDING_MODELS
```
