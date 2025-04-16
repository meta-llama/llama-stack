# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# (Required) NeMo Microservices URLs
NDS_URL = "https://datastore.int.aire.nvidia.com" # Data Store
NEMO_URL = "https://nmp.int.aire.nvidia.com" # Customizer, Evaluator, Guardrails
NIM_URL = "https://nim.int.aire.nvidia.com" # NIM

# (Required) Hugging Face Token
HF_TOKEN = ""

# (Optional) Namespace to associate with Datasets and Customization jobs
NAMESPACE = "nvidia-e2e-tutorial"

# (Optional) User ID to associate with Customization jobs - this is currently unused
USER_ID = ""

# (Optional) Project ID to associate with Datasets and Customization jobs
PROJECT_ID = ""

# (Optional) Directory used by Customized to save output model
CUSTOMIZED_MODEL_DIR = "nvidia-e2e-tutorial/test-llama-stack@v1"
