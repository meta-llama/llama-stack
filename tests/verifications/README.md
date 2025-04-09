# Llama Stack Verifications

Llama Stack Verifications provide standardized test suites to ensure API compatibility and behavior consistency across different LLM providers. These tests help verify that different models and providers implement the expected interfaces and behaviors correctly.

## Overview

This framework allows you to run the same set of verification tests against different LLM providers'  OpenAI-compatible endpoints (Fireworks, Together, Groq, Cerebras, etc., and OpenAI itself) to ensure they meet the expected behavior and interface standards.

## Features

The verification suite currently tests:

- Basic chat completions (streaming and non-streaming)
- Image input capabilities
- Structured JSON output formatting
- Tool calling functionality

## Running Tests

To run the verification tests, use pytest with the following parameters:

```bash
cd llama-stack
pytest tests/verifications/openai --provider=<provider-name>
```

Example:
```bash
# Run all tests
pytest tests/verifications/openai --provider=together

# Only run tests with Llama 4 models
pytest tests/verifications/openai --provider=together -k 'Llama-4'
```

### Parameters

- `--provider`: The provider name (openai, fireworks, together, groq, cerebras, etc.)
- `--base-url`: The base URL for the provider's API (optional - defaults to the standard URL for the specified provider)
- `--api-key`: Your API key for the provider (optional - defaults to the standard API_KEY name for the specified provider)

## Supported Providers

The verification suite currently supports:
- OpenAI
- Fireworks
- Together
- Groq
- Cerebras

## Adding New Test Cases

To add new test cases, create appropriate JSON files in the `openai/fixtures/test_cases/` directory following the existing patterns.


## Structure

- `__init__.py` - Marks the directory as a Python package
- `conftest.py` - Global pytest configuration and fixtures
- `openai/` - Tests specific to OpenAI-compatible APIs
  - `fixtures/` - Test fixtures and utilities
    - `fixtures.py` - Provider-specific fixtures
    - `load.py` - Utilities for loading test cases
    - `test_cases/` - JSON test case definitions
  - `test_chat_completion.py` - Tests for chat completion APIs
