# External Providers

Llama Stack supports external providers that live outside of the main codebase. This allows you to:
- Create and maintain your own providers independently
- Share providers with others without contributing to the main codebase
- Keep provider-specific code separate from the core Llama Stack code

## Configuration

To enable external providers, you need to configure the `external_providers_dir` in your Llama Stack configuration. This directory should contain your external provider specifications:

```yaml
external_providers_dir: /etc/llama-stack/providers.d/
```

## Directory Structure

The external providers directory should follow this structure:

```
providers.d/
  inference/
    custom_ollama.yaml
    vllm.yaml
  vector_io/
    qdrant.yaml
  safety/
    llama-guard.yaml
```

Each YAML file in these directories defines a provider specification for that particular API.

## Provider Specification

A provider specification is a YAML file that contains:
- The adapter configuration
- Required dependencies
- API dependencies

Here's an example for a custom Ollama provider:

```yaml
adapter:
  adapter_type: custom_ollama
  pip_packages: ["ollama", "aiohttp"]
  config_class: llama_stack_ollama_provider.config.OllamaImplConfig
  module: llama_stack_ollama_provider
api_dependencies: []
optional_api_dependencies: []
```

### Adapter Configuration

The `adapter` section defines how to load and configure the provider:

- `adapter_type`: A unique identifier for this adapter
- `pip_packages`: List of Python packages required by the provider
- `config_class`: The full path to the configuration class
- `module`: The Python module containing the provider implementation

### Dependencies

- `api_dependencies`: List of Llama Stack APIs that this provider depends on
- `optional_api_dependencies`: List of optional Llama Stack APIs that this provider can use

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

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
packages = ["src/llama_stack_provider_ollama"]
```

3. Create the provider specification:

```yaml
# /etc/llama-stack/providers.d/inference/custom_ollama.yaml
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
external_providers_dir: /etc/llama-stack/providers.d/
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

## Troubleshooting

If your external provider isn't being loaded:

1. Check that the `external_providers_dir` path is correct and accessible
2. Verify that the YAML files are properly formatted
3. Ensure all required Python packages are installed
4. Check the Llama Stack serverlogs for any error messages
5. Verify that the provider package is installed in your Python environment
