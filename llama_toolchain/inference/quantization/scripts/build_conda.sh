#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Error: Please provide the name of CONDA environment you wish to create"
    exit 1
fi

ENV_NAME=$1

set -eu
eval "$(conda shell.bash hook)"

echo "Will build env (or overwrite) named '$ENV_NAME'"

set -x

run_build() {
    # Set up the conda environment
    yes | conda remove --name $ENV_NAME --all
    yes | conda create -n $ENV_NAME python=3.10
    conda activate $ENV_NAME

    # PT nightly
    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121

    # install dependencies for `llama-agentic-system`
    pip install -r fp8_requirements.txt
}

run_build
