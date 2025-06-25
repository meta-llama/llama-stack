# Llama Stack Unit Tests

You can run the unit tests by running:

```bash
source .venv/bin/activate
./scripts/unit-tests.sh [PYTEST_ARGS]
```

Any additional arguments are passed to pytest. For example, you can specify a test directory, a specific test file, or any pytest flags (e.g., -vvv for verbosity). If no test directory is specified, it defaults to "tests/unit", e.g:

```bash
./scripts/unit-tests.sh tests/unit/registry/test_registry.py -vvv
```

If you'd like to run for a non-default version of Python (currently 3.12), pass `PYTHON_VERSION` variable as follows:

```
source .venv/bin/activate
PYTHON_VERSION=3.13 ./scripts/unit-tests.sh
```
