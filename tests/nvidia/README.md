# NVIDIA tests

## Running tests

**Install the required dependencies:**
    ```bash
    pip install pytest pytest-asyncio pytest-httpx
    ```

There are three modes for testing:

1. Unit tests - this mode checks the provider functionality and does not require a network connection or running distribution

    ```bash
    pytest tests/nvidia/unit
    ```

2. Integration tests against hosted preview APIs - this mode checks the provider functionality against a live system and requires an API key. Get an API key by 0. going to https://build.nvidia.com, 1. selecting a Llama model, e.g. https://build.nvidia.com/meta/llama-3_1-8b-instruct, and 2. clicking "Get API Key". Store the API key in the `NVIDIA_API_KEY` environment variable.

    ```bash
    export NVIDIA_API_KEY=...

    pytest tests/nvidia/integration --base-url https://integrate.api.nvidia.com
    ```

3. Integration tests against a running distribution - this mode checks the provider functionality in the context of a running distribution. This involves running a local NIM, see https://build.nvidia.com/meta/llama-3_1-8b-instruct?snippet_tab=Docker, and creating & configuring a distribution to use it. Details to come.
