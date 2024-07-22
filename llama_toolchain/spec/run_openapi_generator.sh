#!/bin/bash

set -x

PYTHONPATH=/data/users/rsm/llama-models:/data/users/rsm/llama-toolchain:/data/users/rsm/llama-agentic-system:../../../oss-ops:../.. python -m toolchain.spec.generate
