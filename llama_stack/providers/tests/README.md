# Testing Llama Stack Providers

The Llama Stack is designed as a collection of Lego blocks -- various APIs -- which are composable and can be used to quickly and reliably build an app. We need a testing setup which is relatively flexible to enable easy combinations of these providers.

We use `pytest` and all of its dynamism to enable the features needed. Specifically:

- We use `pytest_addoption` to add CLI options allowing you to override providers, models, etc.

- We use `pytest_generate_tests` to dynamically parametrize our tests. This allows us to support a default set of (providers, models, etc.) combinations but retain the flexibility to override them via the CLI if needed.

- We use `pytest_configure` to make sure we dynamically add appropriate marks based on the fixtures we make.

- We use `pytest_collection_modifyitems` to filter tests based on the test config (if specified).

## Pre-requisites

Your development environment should have been configured as per the instructions in the
[CONTRIBUTING.md](../../../CONTRIBUTING.md) file. In particular, make sure to install the test extra
dependencies. Below is the full configuration:


```bash
$ cd llama-stack
$ uv sync --extra dev --extra test
$ uv pip install -e .
$ source .venv/bin/activate
```

## Common options

All tests support a `--providers` option which can be a string of the form `api1=provider_fixture1,api2=provider_fixture2`. So, when testing safety (which need inference and safety APIs) you can use `--providers inference=together,safety=meta_reference` to use these fixtures in concert.

Depending on the API, there are custom options enabled. For example, `inference` tests allow for an `--inference-model` override, etc.

By default, we disable warnings and enable short tracebacks. You can override them using pytest's flags as appropriate.

Some providers need special API keys or other configuration options to work. You can check out the individual fixtures (located in `tests/<api>/fixtures.py`) for what these keys are. These can be specified using the `--env` CLI option. You can also have it be present in the environment (exporting in your shell) or put it in the `.env` file in the directory from which you run the test. For example, to use the Together fixture you can use `--env TOGETHER_API_KEY=<...>`

## Inference

We have the following orthogonal parametrizations (pytest "marks") for inference tests:
- providers: (meta_reference, together, fireworks, ollama)
- models: (llama_8b, llama_3b)

If you want to run a test with the llama_8b model with fireworks, you can use:
```bash
pytest -s -v llama_stack/providers/tests/inference/test_text_inference.py \
  -m "fireworks and llama_8b" \
  --env FIREWORKS_API_KEY=<...>
```

You can make it more complex to run both llama_8b and llama_3b on Fireworks, but only llama_3b with Ollama:
```bash
pytest -s -v llama_stack/providers/tests/inference/test_text_inference.py \
  -m "fireworks or (ollama and llama_3b)" \
  --env FIREWORKS_API_KEY=<...>
```

Finally, you can override the model completely by doing:
```bash
pytest -s -v llama_stack/providers/tests/inference/test_text_inference.py \
  -m fireworks \
  --inference-model "meta-llama/Llama3.1-70B-Instruct" \
  --env FIREWORKS_API_KEY=<...>
```

> [!TIP]
> If you’re using `uv`, you can isolate test executions by prefixing all commands with `uv run pytest...`.

## Agents

The Agents API composes three other APIs underneath:
- Inference
- Safety
- Memory

Given that each of these has several fixtures each, the set of combinations is large. We provide a default set of combinations (see `tests/agents/conftest.py`) with easy to use "marks":
- `meta_reference` -- uses all the `meta_reference` fixtures for the dependent APIs
- `together` -- uses Together for inference, and `meta_reference` for the rest
- `ollama` -- uses Ollama for inference, and `meta_reference` for the rest

An example test with Together:
```bash
pytest -s -m together llama_stack/providers/tests/agents/test_agents.py  \
 --env TOGETHER_API_KEY=<...>
 ```

If you want to override the inference model or safety model used, you can use the `--inference-model` or `--safety-shield` CLI options as appropriate.

If you wanted to test a remotely hosted stack, you can use `-m remote` as follows:
```bash
pytest -s -m remote llama_stack/providers/tests/agents/test_agents.py \
  --env REMOTE_STACK_URL=<...>
```

## Test Config
If you want to run a test suite with a custom set of tests and parametrizations, you can define a YAML test config under llama_stack/providers/tests/ folder and pass the filename through `--config` option as follows:

```
pytest llama_stack/providers/tests/ --config=ci_test_config.yaml
```

### Test config format
Currently, we support test config on inference, agents and memory api tests.

Example format of test config can be found in ci_test_config.yaml.
