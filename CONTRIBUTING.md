# Contributing to Llama-Stack
We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

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


## Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to run linting and formatting checks on your code. You can install the pre-commit hooks by running:

```bash
$ cd llama-stack
$ conda activate <your-environment>
$ pip install pre-commit
$ pre-commit install
```

After that, pre-commit hooks will run automatically before each commit.


## Coding Style
* 2 spaces for indentation rather than tabs
* 80 character line length
* ...

## Common Tasks

Some tips about common tasks you work on while contributing to Llama Stack:

### Using `llama stack build`

Building a stack image (conda / docker) will use the production version of the `llama-stack`, `llama-models` and `llama-stack-client` packages. If you are developing with a llama-stack repository checked out and need your code to be reflected in the stack image, set `LLAMA_STACK_DIR` and `LLAMA_MODELS_DIR` to the appropriate checked out directories when running any of the `llama` CLI commands.

Example:
```bash
$ cd work/
$ git clone https://github.com/meta-llama/llama-stack.git
$ git clone https://github.com/meta-llama/llama-models.git
$ cd llama-stack
$ LLAMA_STACK_DIR=$(pwd) LLAMA_MODELS_DIR=../llama-models llama stack build --template <...>
```


### Updating Provider Configurations

If you have made changes to a provider's configuration in any form (introducing a new config key, or changing models, etc.), you should run `python llama_stack/scripts/distro_codegen.py` to re-generate various YAML files as well as the documentation. You should not change `docs/source/.../distributions/` files manually as they are auto-generated.

### Building the Documentation

If you are making changes to the documentation at [https://llama-stack.readthedocs.io/en/latest/](https://llama-stack.readthedocs.io/en/latest/), you can use the following command to build the documentation and preview your changes. You will need [Sphinx](https://www.sphinx-doc.org/en/master/) and the readthedocs theme.

```bash
cd llama-stack/docs
pip install -r requirements.txt
pip install sphinx-autobuild

# This will start a local server (usually at http://127.0.0.1:8000) that automatically rebuilds and refreshes when you make changes to the documentation.
make html
sphinx-autobuild source build/html
```


## License
By contributing to Llama, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
