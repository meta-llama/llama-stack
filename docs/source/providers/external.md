# External Providers

Llama Stack supports external providers that live outside of the main codebase. This allows you to:
- Create and maintain your own providers independently
- Share providers with others without contributing to the main codebase
- Keep provider-specific code separate from the core Llama Stack code

## Configuration

To enable external providers, you need to configure the `external_providers_dir` in your Llama Stack configuration. This directory should contain your external provider specifications:

```yaml
external_providers_dir: ~/.llama/providers.d/
```

## Directory Structure

The external providers directory should follow this structure:

```
providers.d/
  remote/
    inference/
      custom_ollama.yaml
      vllm.yaml
    vector_io/
      qdrant.yaml
    safety/
      llama-guard.yaml
  inline/
    inference/
      custom_ollama.yaml
      vllm.yaml
    vector_io/
      qdrant.yaml
    safety/
      llama-guard.yaml
```

Each YAML file in these directories defines a provider specification for that particular API.

## Provider Types

Llama Stack supports two types of external providers:

1. **Remote Providers**: Providers that communicate with external services (e.g., cloud APIs)
2. **Inline Providers**: Providers that run locally within the Llama Stack process

## Known External Providers

Here's a list of known external providers that you can use with Llama Stack:

| Name | Description | API | Type | Repository |
|------|-------------|-----|------|------------|
| KubeFlow Training | Train models with KubeFlow | Post Training | Remote | [llama-stack-provider-kft](https://github.com/opendatahub-io/llama-stack-provider-kft) |
| KubeFlow Pipelines | Train models with KubeFlow Pipelines | Post Training | Inline **and** Remote | [llama-stack-provider-kfp-trainer](https://github.com/opendatahub-io/llama-stack-provider-kfp-trainer) |
| RamaLama | Inference models with RamaLama | Inference | Remote | [ramalama-stack](https://github.com/containers/ramalama-stack) |
| TrustyAI LM-Eval | Evaluate models with TrustyAI LM-Eval | Eval | Remote | [llama-stack-provider-lmeval](https://github.com/trustyai-explainability/llama-stack-provider-lmeval) |

### Remote Provider Specification

Remote providers are used when you need to communicate with external services. Here's an example for a custom Ollama provider:

```yaml
adapter:
  adapter_type: custom_ollama
  pip_packages:
  - ollama
  - aiohttp
  config_class: llama_stack_ollama_provider.config.OllamaImplConfig
  module: llama_stack_ollama_provider
api_dependencies: []
optional_api_dependencies: []
```

#### Adapter Configuration

The `adapter` section defines how to load and configure the provider:

- `adapter_type`: A unique identifier for this adapter
- `pip_packages`: List of Python packages required by the provider
- `config_class`: The full path to the configuration class
- `module`: The Python module containing the provider implementation

### Inline Provider Specification

Inline providers run locally within the Llama Stack process. Here's an example for a custom vector store provider:

```yaml
module: llama_stack_vector_provider
config_class: llama_stack_vector_provider.config.VectorStoreConfig
pip_packages:
  - faiss-cpu
  - numpy
api_dependencies:
  - inference
optional_api_dependencies:
  - vector_io
provider_data_validator: llama_stack_vector_provider.validator.VectorStoreValidator
container_image: custom-vector-store:latest  # optional
```

#### Inline Provider Fields

- `module`: The Python module containing the provider implementation
- `config_class`: The full path to the configuration class
- `pip_packages`: List of Python packages required by the provider
- `api_dependencies`: List of Llama Stack APIs that this provider depends on
- `optional_api_dependencies`: List of optional Llama Stack APIs that this provider can use
- `provider_data_validator`: Optional validator for provider data
- `container_image`: Optional container image to use instead of pip packages

## Required Implementation

### Remote Providers

Remote providers must expose a `get_adapter_impl()` function in their module that takes two arguments:
1. `config`: An instance of the provider's config class
2. `deps`: A dictionary of API dependencies

This function must return an instance of the provider's adapter class that implements the required protocol for the API.

Example:
```python
async def get_adapter_impl(
    config: OllamaImplConfig, deps: Dict[Api, Any]
) -> OllamaInferenceAdapter:
    return OllamaInferenceAdapter(config)
```

### Inline Providers

Inline providers must expose a `get_provider_impl()` function in their module that takes two arguments:
1. `config`: An instance of the provider's config class
2. `deps`: A dictionary of API dependencies

Example:
```python
async def get_provider_impl(
    config: VectorStoreConfig, deps: Dict[Api, Any]
) -> VectorStoreImpl:
    impl = VectorStoreImpl(config, deps[Api.inference])
    await impl.initialize()
    return impl
```

## Dependencies

The provider package must be installed on the system. For example:

```bash
$ uv pip show llama-stack-ollama-provider
Name: llama-stack-ollama-provider
Version: 0.1.0
Location: /path/to/venv/lib/python3.10/site-packages
```

## Example: Custom Ollama Provider

Here's a complete example of creating and using a custom Ollama provider:

1. First, create the provider package:

```bash
mkdir -p llama-stack-provider-ollama
cd llama-stack-provider-ollama
git init
uv init
```

2. Edit `pyproject.toml`:

```toml
[project]
name = "llama-stack-provider-ollama"
version = "0.1.0"
description = "Ollama provider for Llama Stack"
requires-python = ">=3.10"
dependencies = ["llama-stack", "pydantic", "ollama", "aiohttp"]
```

3. Create the provider specification:

```yaml
# ~/.llama/providers.d/remote/inference/custom_ollama.yaml
adapter:
  adapter_type: custom_ollama
  pip_packages: ["ollama", "aiohttp"]
  config_class: llama_stack_provider_ollama.config.OllamaImplConfig
  module: llama_stack_provider_ollama
api_dependencies: []
optional_api_dependencies: []
```

4. Install the provider:

```bash
uv pip install -e .
```

5. Configure Llama Stack to use external providers:

```yaml
external_providers_dir: ~/.llama/providers.d/
```

The provider will now be available in Llama Stack with the type `remote::custom_ollama`.

## Best Practices

1. **Package Naming**: Use the prefix `llama-stack-provider-` for your provider packages to make them easily identifiable.

2. **Version Management**: Keep your provider package versioned and compatible with the Llama Stack version you're using.

3. **Dependencies**: Only include the minimum required dependencies in your provider package.

4. **Documentation**: Include clear documentation in your provider package about:
   - Installation requirements
   - Configuration options
   - Usage examples
   - Any limitations or known issues

5. **Testing**: Include tests in your provider package to ensure it works correctly with Llama Stack.
You can refer to the [integration tests
guide](https://github.com/meta-llama/llama-stack/blob/main/tests/integration/README.md) for more
information. Execute the test for the Provider type you are developing.

## Troubleshooting

If your external provider isn't being loaded:

1. Check that the `external_providers_dir` path is correct and accessible.
2. Verify that the YAML files are properly formatted.
3. Ensure all required Python packages are installed.
4. Check the Llama Stack server logs for any error messages - turn on debug logging to get more
   information using `LLAMA_STACK_LOGGING=all=debug`.
5. Verify that the provider package is installed in your Python environment.
