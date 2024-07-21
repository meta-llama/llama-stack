#!/bin/bash

# Function to prompt the user for input and set the variables
read_input() {
    read -p "Enter the checkpoint directory (e.g., /home/dalton/models/Meta-Llama-3.1-8B-Instruct-20240710150000): " checkpoint_dir
    read -p "Enter the model parallel size (e.g., 1): " model_parallel_size
}

# Function to expand paths starting with "~/" to full paths
expand_path() {
    local path="$1"
    if [[ "$path" == "~/"* ]]; then
        echo "$HOME/${path:2}"
    else
        echo "$path"
    fi
}

# Function to create parent directory if it does not exist
create_parent_dir() {
    local parent_dir=$(dirname "${yaml_output_path}")
    if [ ! -d "${parent_dir}" ]; then
        mkdir -p "${parent_dir}"
        echo "Created parent directory: ${parent_dir}"
    fi
}

# Function to output the YAML configuration
output_yaml() {
    cat <<EOL > ${yaml_output_path}
model_inference_config:
  impl_type: "inline"
  inline_config:
    checkpoint_type: "pytorch"
    checkpoint_dir: ${checkpoint_dir}
    tokenizer_path: ${checkpoint_dir}/tokenizer.model
    model_parallel_size: ${model_parallel_size}
    max_seq_len: 2048
    max_batch_size: 1
EOL
    echo "YAML configuration has been written to ${yaml_output_path}"
}

# Main script execution
read_input

# Expand paths
checkpoint_dir=$(expand_path "$checkpoint_dir")

# Define output path
yaml_output_path="toolchain/configs/${USER}.yaml"

create_parent_dir
output_yaml
