# Llama Stack Verifications

Llama Stack Verifications provide standardized test suites to ensure API compatibility and behavior consistency across different LLM providers. These tests help verify that different models and providers implement the expected interfaces and behaviors correctly.

## Overview

This framework allows you to run the same set of verification tests against different LLM providers' OpenAI-compatible endpoints (Fireworks, Together, Groq, Cerebras, etc., and OpenAI itself) to ensure they meet the expected behavior and interface standards.

## Features

The verification suite currently tests the following in both streaming and non-streaming modes:

- Basic chat completions
- Image input capabilities
- Structured JSON output formatting
- Tool calling functionality

## Report

The lastest report can be found at [REPORT.md](REPORT.md).

To update the report, ensure you have the API keys set,
```bash
export OPENAI_API_KEY=<your_openai_api_key>
export FIREWORKS_API_KEY=<your_fireworks_api_key>
export TOGETHER_API_KEY=<your_together_api_key>
```
then run
```bash
uv run python tests/verifications/generate_report.py --run-tests
```

## Running Tests

To run the verification tests, use pytest with the following parameters:

```bash
cd llama-stack
pytest tests/verifications/openai_api --provider=<provider-name>
```

Example:
```bash
# Run all tests
pytest tests/verifications/openai_api --provider=together

# Only run tests with Llama 4 models
pytest tests/verifications/openai_api --provider=together -k 'Llama-4'
```

### Parameters

- `--provider`: The provider name (openai, fireworks, together, groq, cerebras, etc.)
- `--base-url`: The base URL for the provider's API (optional - defaults to the standard URL for the specified provider)
- `--api-key`: Your API key for the provider (optional - defaults to the standard API_KEY name for the specified provider)

## Supported Providers

The verification suite supports any provider with an OpenAI compatible endpoint.

See `tests/verifications/conf/` for the list of supported providers.

To run on a new provider, simply add a new yaml file to the `conf/` directory with the provider config. See `tests/verifications/conf/together.yaml` for an example.

## Adding New Test Cases

To add new test cases, create appropriate JSON files in the `openai_api/fixtures/test_cases/` directory following the existing patterns.


## Structure

- `__init__.py` - Marks the directory as a Python package
- `conf/` - Provider-specific configuration files
- `openai_api/` - Tests specific to OpenAI-compatible APIs
  - `fixtures/` - Test fixtures and utilities
    - `fixtures.py` - Provider-specific fixtures
    - `load.py` - Utilities for loading test cases
    - `test_cases/` - JSON test case definitions
  - `test_chat_completion.py` - Tests for chat completion APIs
