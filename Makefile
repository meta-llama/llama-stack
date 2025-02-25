ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
OS := linux
ifeq ($(shell uname -s), Darwin)
	OS = osx
endif

PYTHON_VERSION = ${shell python --version | grep -Eo '[0-9]\.[0-9]+'}
PYTHON_VERSIONS := 3.10 3.11

build-dev:
	uv sync --extra dev --extra test
	uv pip install -e .
	. .venv/bin/activate
	uv pip install sqlite-vec chardet datasets sentence_transformers pypdf

build-ollama: fix-line-endings
	llama stack build --template ollama --image-type venv

fix-line-endings:
	sed -i '' 's/\r$$//' llama_stack/distribution/common.sh
	sed -i '' 's/\r$$//' llama_stack/distribution/build_venv.sh

test-sqlite-vec:
	pytest llama_stack/providers/tests/vector_io/test_sqlite_vec.py \
	-v -s --tb=short --disable-warnings --asyncio-mode=auto

test-ollama-vector-integration:
	INFERENCE_MODEL=llama3.2:3b-instruct-fp16 LLAMA_STACK_CONFIG=ollama \
	pytest -s -v tests/client-sdk/vector_io/test_vector_io.py


make serve-ollama:
	ollama run llama3.2:3b-instruct-fp16 --keepalive 24h
