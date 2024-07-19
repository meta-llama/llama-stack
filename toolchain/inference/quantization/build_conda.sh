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
    # Set CUDA 9.0a targets
    export CUDA_ARCH_LIST="8.0;9.0a"
    export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90a,code=sm_90a"
    export TORCH_CUDA_ARCH_LIST=$CUDA_ARCH_LIST

    # Set up the conda environment
    yes | conda remove --name $ENV_NAME --all
    yes | conda create -n $ENV_NAME python=3.10
    conda activate $ENV_NAME
    yes | conda install --channel "nvidia/label/cuda-12.1.0" cuda
    yes | conda install cuda-nvtx cuda-nvtx-dev conda-forge::nccl


    # ############# Hack to get CUDA path #############
    ln -s $CONDA_PREFIX/targets/x86_64-linux/include/* $CONDA_PREFIX/include/ || true
    export CUDA_HOME=$CONDA_PREFIX
    export CUDA_BIN_PATH=$CUDA_HOME
    # #################################################

    # PT nightly
    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
    pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu121

    # install dependencies for `llama-agentic-system`
    pip install -r fp8_requirements.txt
}

run_build
