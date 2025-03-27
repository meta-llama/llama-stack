# Contributing to Llama-Stack
We want to make contributing to this project as easy and transparent as
possible.

## Discussions -> Issues -> Pull Requests

We actively welcome your pull requests. However, please read the following. This is heavily inspired by [Ghostty](https://github.com/ghostty-org/ghostty/blob/main/CONTRIBUTING.md).

If in doubt, please open a [discussion](https://github.com/meta-llama/llama-stack/discussions); we can always convert that to an issue later.

**I'd like to contribute!**

All issues are actionable (please report if they are not.) Pick one and start working on it. Thank you.
If you need help or guidance, comment on the issue. Issues that are extra friendly to new contributors are tagged with "contributor friendly".

**I have a bug!**

1. Search the issue tracker and discussions for similar issues.
2. If you don't have steps to reproduce, open a discussion.
3. If you have steps to reproduce, open an issue.

**I have an idea for a feature!**

1. Open a discussion.

**I've implemented a feature!**

1. If there is an issue for the feature, open a pull request.
2. If there is no issue, open a discussion and link to your branch.

**I have a question!**

1. Open a discussion or use [Discord](https://discord.gg/llama-stack).


**Opening a Pull Request**

1. Fork the repo and create your branch from `main`.
2. If you've changed APIs, update the documentation.
3. Ensure the test suite passes.
4. Make sure your code lints using `pre-commit`.
5. If you haven't already, complete the Contributor License Agreement ("CLA").
6. Ensure your pull request follows the [conventional commits format](https://www.conventionalcommits.org/en/v1.0.0/).

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](http://facebook.com/whitehat/info) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.


## Set up your development environment

We use [uv](https://github.com/astral-sh/uv) to manage python dependencies and virtual environments.
You can install `uv` by following this [guide](https://docs.astral.sh/uv/getting-started/installation/).

You can install the dependencies by running:

```bash
cd llama-stack
uv sync --extra dev
uv pip install -e .
source .venv/bin/activate
```

> [!NOTE]
> You can pin a specific version of Python to use for `uv` by adding a `.python-version` file in the root project directory.
> Otherwise, `uv` will automatically select a Python version according to the `requires-python` section of the `pyproject.toml`.
> For more info, see the [uv docs around Python versions](https://docs.astral.sh/uv/concepts/python-versions/).

Note that you can create a dotenv file `.env` that includes necessary environment variables:
```
LLAMA_STACK_BASE_URL=http://localhost:8321
LLAMA_STACK_CLIENT_LOG=debug
LLAMA_STACK_PORT=8321
LLAMA_STACK_CONFIG=<provider-name>
TAVILY_SEARCH_API_KEY=
BRAVE_SEARCH_API_KEY=
```

And then use this dotenv file when running client SDK tests via the following:
```bash
uv run --env-file .env -- pytest -v tests/integration/inference/test_text_inference.py
```

## Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to run linting and formatting checks on your code. You can install the pre-commit hooks by running:

```bash
uv run pre-commit install
```

After that, pre-commit hooks will run automatically before each commit.

Alternatively, if you don't want to install the pre-commit hooks, you can run the checks manually by running:

```bash
uv run pre-commit run --all-files
```

> [!CAUTION]
> Before pushing your changes, make sure that the pre-commit hooks have passed successfully.

## Running unit tests

You can run the unit tests by running:

```bash
source .venv/bin/activate
./scripts/unit-tests.sh
```

If you'd like to run for a non-default version of Python (currently 3.10), pass `PYTHON_VERSION` variable as follows:

```
source .venv/bin/activate
PYTHON_VERSION=3.13 ./scripts/unit-tests.sh
```

## Adding a new dependency to the project

To add a new dependency to the project, you can use the `uv` command. For example, to add `foo` to the project, you can run:

```bash
uv add foo
uv sync
```

## Coding Style

* Comments should provide meaningful insights into the code. Avoid filler comments that simply describe the next step, as they create unnecessary clutter, same goes for docstrings.
* Prefer comments to clarify surprising behavior and/or relationships between parts of the code rather than explain what the next line of code does.
* Catching exceptions, prefer using a specific exception type rather than a broad catch-all like `Exception`.
* Error messages should be prefixed with "Failed to ..."
* 4 spaces for indentation rather than tabs

## Common Tasks

Some tips about common tasks you work on while contributing to Llama Stack:

### Using `llama stack build`

Building a stack image (conda / docker) will use the production version of the `llama-stack` and `llama-stack-client` packages. If you are developing with a llama-stack repository checked out and need your code to be reflected in the stack image, set `LLAMA_STACK_DIR` and `LLAMA_STACK_CLIENT_DIR` to the appropriate checked out directories when running any of the `llama` CLI commands.

Example:
```bash
cd work/
git clone https://github.com/meta-llama/llama-stack.git
git clone https://github.com/meta-llama/llama-stack-client-python.git
cd llama-stack
LLAMA_STACK_DIR=$(pwd) LLAMA_STACK_CLIENT_DIR=../llama-stack-client-python llama stack build --template <...>
```


### Updating Provider Configurations

If you have made changes to a provider's configuration in any form (introducing a new config key, or changing models, etc.), you should run `./scripts/distro_codegen.py` to re-generate various YAML files as well as the documentation. You should not change `docs/source/.../distributions/` files manually as they are auto-generated.

### Building the Documentation

If you are making changes to the documentation at [https://llama-stack.readthedocs.io/en/latest/](https://llama-stack.readthedocs.io/en/latest/), you can use the following command to build the documentation and preview your changes. You will need [Sphinx](https://www.sphinx-doc.org/en/master/) and the readthedocs theme.

```bash
cd docs
uv sync --extra docs

# This rebuilds the documentation pages.
uv run make html

# This will start a local server (usually at http://127.0.0.1:8000) that automatically rebuilds and refreshes when you make changes to the documentation.
uv run sphinx-autobuild source build/html --write-all
```

### Update API Documentation

If you modify or add new API endpoints, update the API documentation accordingly. You can do this by running the following command:

```bash
uv run --with ".[dev]" ./docs/openapi_generator/run_openapi_generator.sh
```

The generated API documentation will be available in `docs/_static/`. Make sure to review the changes before committing.

## License
By contributing to Llama, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
