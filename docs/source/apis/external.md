# External APIs

Llama Stack supports external APIs that live outside of the main codebase. This allows you to:
- Create and maintain your own APIs independently
- Share APIs with others without contributing to the main codebase
- Keep API-specific code separate from the core Llama Stack code

## Configuration

To enable external APIs, you need to configure the `external_apis_dir` in your Llama Stack configuration. This directory should contain your external API specifications:

```yaml
external_apis_dir: ~/.llama/apis.d/
```

## Directory Structure

The external APIs directory should follow this structure:

```
apis.d/
  custom_api1.yaml
  custom_api2.yaml
```

Each YAML file in these directories defines an API specification.

## API Specification

Here's an example of an external API specification for a weather API:

```yaml
module: weather
api_dependencies:
  - inference
route_prefix: /weather
protocol: WeatherAPI
name: weather
pip_packages:
  - llama-stack-api-weather
```

### API Specification Fields

- `module`: Python module containing the API implementation
- `route_prefix`: URL prefix for the API routes
- `protocol`: Name of the protocol class for the API
- `name`: Name of the API
- `pip_packages`: List of pip packages to install the API, typically a single package

## Required Implementation

External APIs must expose a `available_providers()` function in their module that returns a list of provider names:

```python
# llama_stack_api_weather/api.py
from llama_stack.providers.datatypes import Api, InlineProviderSpec, ProviderSpec


def available_providers() -> list[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.weather,
            provider_type="inline::darksky",
            pip_packages=[],
            module="llama_stack_provider_darksky",
            config_class="llama_stack_provider_darksky.DarkSkyWeatherImplConfig",
        ),
    ]
```

A Protocol class like so:

```python
# llama_stack_api_weather/api.py
from typing import Protocol

from llama_stack.schema_utils import webmethod


class WeatherAPI(Protocol):
    """
    A protocol for the Weather API.
    """

    @webmethod(route="/locations", method="GET")
    async def get_available_locations() -> dict[str, list[str]]:
        """
        Get the available locations.
        """
        ...
```

## Example: Custom API

Here's a complete example of creating and using a custom API:

1. First, create the API package:

```bash
mkdir -p llama-stack-api-weather
cd llama-stack-api-weather
git init
uv init
```

2. Edit `pyproject.toml`:

```toml
[project]
name = "llama-stack-api-weather"
version = "0.1.0"
description = "Weather API for Llama Stack"
requires-python = ">=3.10"
dependencies = ["llama-stack", "pydantic"]
```

3. Create the API implementation:

```python
# llama_stack_api_weather/api.py
from typing import Protocol

from llama_stack.providers.datatypes import Api, InlineProviderSpec, ProviderSpec
from llama_stack.schema_utils import webmethod


def available_providers() -> list[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.weather,
            provider_type="inline::darksky",
            pip_packages=[],
            module="llama_stack_provider_darksky",
            config_class="llama_stack_provider_darksky.DarkSkyWeatherImplConfig",
        ),
    ]


class WeatherAPI(Protocol):
    """
    A protocol for the Weather API.
    """

    @webmethod(route="/locations", method="GET")
    async def get_available_locations() -> dict[str, list[str]]:
        """
        Get the available locations.
        """
        ...
```

4. Create the API specification:

```yaml
# ~/.llama/apis.d/weather.yaml
module: llama_stack_api_weather.api
name: weather
pip_packages: ["llama-stack-api-weather"]
route_prefix: /weather
protocol: WeatherAPI

```

5. Install the API package:

```bash
uv pip install -e .
```

6. Configure Llama Stack to use external APIs:

```yaml
version: "2"
image_name: "llama-stack-api-weather"
apis:
  - weather
providers: {}
external_apis_dir: ~/.llama/apis.d
```

The API will now be available at `/v1/weather/locations`.

## Best Practices

1. **Package Naming**: Use a clear and descriptive name for your API package.

2. **Version Management**: Keep your API package versioned and compatible with the Llama Stack version you're using.

3. **Dependencies**: Only include the minimum required dependencies in your API package.

4. **Documentation**: Include clear documentation in your API package about:
   - Installation requirements
   - Configuration options
   - API endpoints and usage
   - Any limitations or known issues

5. **Testing**: Include tests in your API package to ensure it works correctly with Llama Stack.

## Troubleshooting

If your external API isn't being loaded:

1. Check that the `external_apis_dir` path is correct and accessible.
2. Verify that the YAML files are properly formatted.
3. Ensure all required Python packages are installed.
4. Check the Llama Stack server logs for any error messages - turn on debug logging to get more information using `LLAMA_STACK_LOGGING=all=debug`.
5. Verify that the API package is installed in your Python environment.
