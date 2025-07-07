# Testing Llama Stack

Tests are of three different kinds:
- Unit tests
- Provider focused integration tests
- Client SDK tests

## Unit Tests

Unit tests verify individual components and functions in isolation. They are fast, reliable, and don't require external services.

### Prerequisites

1. **Python Environment**: Ensure you have Python 3.12+ installed
2. **uv Package Manager**: Install `uv` if not already installed
3. **Virtual Environment**: Activate the project's virtual environment

### Running Unit Tests

#### Using the Unit Test Script (Recommended)

The easiest way to run unit tests is using the provided script:

```bash
# Activate virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Run all unit tests and generate coverage report
PYTHON_VERSION=3.12 ./scripts/unit-tests.sh
```

### Test Configuration

- **Test Discovery**: Tests are automatically discovered in the `tests/unit/` directory
- **Async Support**: Tests use `--asyncio-mode=auto` for automatic async test handling
- **Coverage**: Tests generate coverage reports in `htmlcov/` directory
- **Python Version**: Defaults to Python 3.12, but can be overridden with `PYTHON_VERSION` environment variable

### Coverage Reports

After running tests, you can view coverage reports:

```bash
# Open HTML coverage report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```
