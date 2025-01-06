import os
import re
import signal
import subprocess
import time

import pytest


# Inference provider and the required environment arg for running integration tests
INFERENCE_PROVIDER_ENV_KEY = {
    "ollama": None,
    "fireworks": "FIREWORKS_API_KEY",
    "together": "TOGETHER_API_KEY",
}

TEST_MODELS = {
    "text": "meta-llama/Llama-3.1-8B-Instruct",
    "vision": "meta-llama/Llama-3.2-11B-Vision-Instruct",
}

# Model category and the keywords of the corresponding functionality tests
CATEGORY_FUNCTIONALITY_TESTS = {
    "text": ["streaming", "tool_calling", "structured_output"],
    "vision": [
        "streaming",
    ],
}


def generate_pytest_args(category, provider, test_keywords, env_key):
    test_path = (
        "./llama_stack/providers/tests/inference/test_{model_type}_inference.py".format(
            model_type=category
        )
    )
    pytest_args = [
        test_path,
        "-v",
        # "-s",
        "-k",
        "{provider} and ({test_keywords})".format(
            provider=provider, test_keywords=" or ".join(test_keywords)
        ),
        "--inference-model={model_name}".format(model_name=TEST_MODELS[category]),
    ]

    if env_key is not None:
        pytest_args.extend(
            [
                "--env",
                "{key_name}={key_value}".format(
                    key_name=env_key, key_value=os.getenv(env_key)
                ),
            ]
        )
    return pytest_args


def main():
    test_result = []

    for model_category, test_keywords in CATEGORY_FUNCTIONALITY_TESTS.items():
        for provider, env_key in INFERENCE_PROVIDER_ENV_KEY.items():
            if provider == "ollama":
                ollama_model_alias = (
                    "llama3.1:8b-instruct-fp16"
                    if model_category == "text"
                    else "llama3.2-vision:11b-instruct-fp16"
                )
                proc = subprocess.Popen(["ollama", "run", ollama_model_alias])
                retcode = pytest.main(
                    generate_pytest_args(
                        model_category, provider, test_keywords, env_key
                    )
                )
                proc = subprocess.Popen(["ollama", "stop", ollama_model_alias])
            else:
                retcode = pytest.main(
                    generate_pytest_args(
                        model_category, provider, test_keywords, env_key
                    )
                )


if __name__ == "__main__":
    main()
