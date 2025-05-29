# Llama Stack Provider Configuration API

**Authors:**

* @cdoern

## Rationale

Currently there is no way for a client to

1. see the current provider configuration
2. edit the current provider configuration after server stand-up

These gaps stood out to me as usability issues. Having the ability for a single user system (and multi user!) to be able to see their configuration and responsibly manipulate it is important.

These two things will make it hard long term for tasks like Synthetic Data Generation (SDG), Training, Evals, etc to be run in a repeatable and knowledgeable manner by end users. Llama Stack of today is built to be a "black box" to not expose sensitive configuration to the user. Exposing the current provider configuration to a user will help them understand what they will be running for various providers as functionality gets more complex (SDG, Evals, Training, etc). Additionally, allowing a user to apply parts of a config on top of a running stack as opposed to taking the stack down and having the admin apply a full run config again seems like a more sustainable workflow.

A concrete example is modification of the served models, the vLLM endpoint, etc: which currently require stopping LLS and vLLM and spinning both back up with new config. this API would enable the LLS to stay up and the provider config be modified to represent the current state of vLLM.

Not having the ability to swap in configuration based on the circumstances at run time will make the potential for hardware detection, hyperparameter application, and the general UX of LLS more difficult than today.

This seems like a table stakes set of operations, and one we will likely find ourselves needing as the feature set expands.

## Solution Overview

1. New API route: /v1/providers
2. Register endpoint for provider configurations: /v1/providers/register
3. Modify provider inspect API to return a provider configuration: /v1/inspect/providers/{provider_id}
4. New resources in the llama-stack-client-python (and the other clients?) to handle Configurations
5. Llama-stack-client commands for configurations
6. Upstream consensus on this new API


## Demo Code Available for Testing

I have been working on this on my forks of LLS and the python client. This API works and enables users to apply configuration and inspect it all with a running server.

To test this out yourself you can look at my custom fork of LLS and the client both from the `config` branch:

How-to: https://github.com/cdoern/llama-stack/blob/config/docs/config_api.md
llama-stack: https://github.com/cdoern/llama-stack/tree/config
llama-stack-client-python: https://github.com/cdoern/llama-stack-client-python/tree/config


## How can someone use this?

Here is a modified version of the SDK example script which changes the provider configuration of a server:


```python
import os
import sys
import json
import yaml


def create_library_client(template="ollama"):
    from llama_stack import LlamaStackAsLibraryClient

    client = LlamaStackAsLibraryClient(template)
    client.initialize()
    return client


client = create_library_client()

prov = client.providers.list()
config = client.providers.inspect()

print("Old Config \n", yaml.dump(config, indent=2))


# put together a new config. Can also be read from a file
# important point here is that this can be a partial provider config list.

config = {
    "inference": [
        {
            "provider_id": "ollama",
            "provider_type": "remote::ollama",
            "config": {"url": "http://localhost:12345"},
        }
    ]
}
config = json.dumps(config)


# register configuration for an existing provider in the stack.
config = client.providers.register(config=config)


# get new configuration

config = client.providers.inspect()
try:
    print("New Configuration \n", yaml.dump(config, indent=2))
except Exception as exc:
    print(f"could not dump yaml: {exc}")


# List available models

try:
    models = client.models.list()
except AttributeError as e:
    print(e)
    sys.exit(1)
print("--- Available models: ---")
for m in models:
    print(f"- {m.identifier}")
print()

# uses the NEW ollama URL not the old one.

response = client.inference.chat_completion(
    model_id=os.environ["INFERENCE_MODEL"],
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about coding"},
    ],
)
```

This script switches the ollama URL of an existing server to point to another ollama server. I used the ollama URL as the first field just because it was easy to test. The actual user configurable fields will at first be minimal just to get this API off the ground but can be easily expanded.  I also intend to expose this API through the llama-stack-client with the following commands:

`llama-stack-client providers inspect PROVIDER_ID`
`llama-stack-client providers register –config-path –config…`

## Some Key Features

### Class Structure

Here is what I have so far in terms of net-new API classes inside of llama-stack:


```python
@json_schema_type
class Configuration(BaseModel):
    type: Literal[ResourceType.configuration.value] = ResourceType.configuration.value
    config: StackRunConfig


@runtime_checkable
@trace_protocol
class Providers(Protocol):
    """Llama Stack Configuration API for storing and applying hyperparameters for given tasks."""

    @webmethod(route="/providers/register", method="POST")
    async def register_provider(
        self,
        config,
    ) -> dict[str, Any]: ...
```

And for /v1/inspect:

```python
@runtime_checkable
class Inspect(Protocol):

    @webmethod(route="/inspect/providers", method="GET")
    async def inspect_config(
        self,
    ) -> InspectProvidersResponse: ...
```

Where this is InspectProvidersResponse:

```python
class InspectProvidersResponse(BaseModel):
    data: Provider | None


class ListProvidersResponse(BaseModel):
    data: List[ProviderInfo]
```

InspectProviderResponse differs from ListProviderResponse. `/v1/inspect/providers` will __list__ providers while `/v1/inspect/providers/{provider_id}` will inspect a specific provider and return its configuration.

And this is a UserConfig: (There is a lot of overlap in methods, this is all hacking code!)

```python
class UserConfig(BaseModel):
    providers: Dict[str, List[Provider]] = Field(
        description="""
One or more providers to use for each API. The same provider_type (e.g., meta-reference)
can be instantiated multiple times (with different configs) if necessary.
""",
    )

    @classmethod
    def from_stack_run(
        cls, registry: Dict[Any, Dict[str, Any]], stack_run: "StackRunConfig"
    ) -> "UserConfig":
        """
        This is almost a method to go backwards and get a user config from an existing run config
        """
        user_config: Dict[str, List[Provider]] = {}
        for type, providers in stack_run.providers.items():
            api = Api(type)
            user_config[type] = []
            for provider in providers:
                provider_config = {}
                provider_spec = registry[api][provider.provider_type]
                config_type = instantiate_class_type(provider_spec.config_class)
                try:
                    if provider.config:
                        existing = config_type(**provider.config)
                        for field_name, field in existing.model_fields.items():
                            if field.json_schema_extra:
                                provider_config[field_name] = getattr(
                                    existing, field_name
                                )
                        user_config[type].append(
                            Provider(
                                provider_id=provider.provider_id,
                                provider_type=provider.provider_type,
                                config=provider_config,
                            )
                        )
                except Exception as exc:
                    print(
                        f"Could not instantiate UserConfig due to improper provider config {exc}"
                    )
        return cls(providers=user_config)

    @classmethod
    def from_providers(
        cls, registry: Dict[Any, Dict[str, Any]], providers: Dict[str, List[Provider]]
    ):
        """
        This is a method to go forward, validate that a dictionary of providers is _only_ a user config
        """
        user_config: Dict[str, List[Provider]] = {}
        for type, provider_list in providers.items():
            api = Api(type)
            user_config[type] = []
            provider_config = {}
            for prov in provider_list:
                prov = Provider(**prov)
                provider_spec = registry[api][prov.provider_type]
                config_type = instantiate_class_type(provider_spec.config_class)
                try:
                    if prov.config is not None:
                        existing = config_type(**prov.config)
                        for field_name, field in existing.model_fields.items():
                            if field.json_schema_extra:
                                provider_config[field_name] = getattr(
                                    existing, field_name
                                )
                            else:
                                print("given configuration is not user configurable.")
                        user_config[type].append(
                            Provider(
                                provider_id=prov.provider_id,
                                provider_type=prov.provider_type,
                                config=provider_config,
                            )
                        )
                except Exception as exc:
                    print(
                        f"Could not instantiate UserConfig due to improper provider config {exc}"
                    )
        return cls(providers=user_config)
```

There are also associated Resources in the client which allow things like `client.providers` to be called without erroring out.

### UserConfig vs StackRunConfig. How do we control what users can and cannot see?

A key part of this API are the fields exposed in both the inspection and registration. A Configuration object contains a StackRunConfig within it. However, the data within this config is a UserConfig. For those unaware, a `StackRunConfig` generally follows the structure of the run.yaml a user specifies when running `llama stack run`.

A UserConfig is a StackRunConfig but only with specific fields displayed to the user and available for modification. Since each provider has its own config class that feeds into the StackRunConfig the following can be used to label certain fields as "User Configurable":

```python
class OllamaImplConfig(BaseModel):
    url: str = Field(DEFAULT_OLLAMA_URL, json_schema_extra={"user_field": True})

    @classmethod
    def sample_run_config(
        cls, url: str = "${env.OLLAMA_URL:http://localhost:11434}", **kwargs
    ) -> Dict[str, Any]:
        return {"url": url}
```

The pydantic json_schema_extra field can then be used to create an intermediary UserConfig before funneling into the returned Configuration object. The User Config will only have fields labeled as user_field meaning that if a user tries to register a configuration with non-user fields specified, they will be dropped, and an inspected configuration will only contain user fields for viewing as well. A user is also only allowed to register and inspect the Providers section of the StackRunConfig meaning they cannot change actual settings of the server, just the provider configurations.

This structure results in `client.providers.inspect()` output like:

```console
 providers:
  agents:
  - config: {}
    provider_id: meta-reference
    provider_type: inline::meta-reference
  datasetio: []
  eval: []
  inference:
  - config:
      url: http://localhost:12345
    provider_id: ollama
    provider_type: remote::ollama
  safety: []
  scoring:
  - config: {}
    provider_id: braintrust
    provider_type: inline::braintrust
  telemetry:
  - config: {}
    provider_id: meta-reference
    provider_type: inline::meta-reference
  tool_runtime:
  - config: {}
    provider_id: brave-search
    provider_type: remote::brave-search
  - config: {}
    provider_id: tavily-search
    provider_type: remote::tavily-search
  vector_io:
  - config: {}
    provider_id: sqlite-vec
    provider_type: inline::sqlite-vec
```

So far, only the ollama URL has been added as a valid UserConfig field, meaning it's the only provider config that shows up.

### How does the registration work?

“Registering” a config does not store an object with a provider as model registration does. Rather, registration of a config re-defines how a provider is configured on the fly. The user of course has the yaml run config to fall back to if they want to reboot the server. However, for all intents and purposes: the run config becomes a combination of what was provided in the register command and the previously existing run configuration.

Functionally this means re-instantiating the endpoint implementations for the entire stack while it is still running. Meaning the client object, if using the SDK, must be refreshed behind the scenes and if using the CLI, the server will have special handling for a `/providers/register/` route being detected before it returns a response to the client.

Additionally, a user cannot register a provider not already initialized in the stack, at least for now. This is because new provider registration would require potential new dependencies to be downloaded which is out of the current scope of this API.

## Alternatives

### User does not need to pass in a provider config object

To further improve UX, rather than passing in a json or yaml object, a user could pass in specific arguments that internally we could parse on the client side before making an API call. Something like:

```python
client.providers.register(
    provider_ids="ollama", configuration={"url": "127.0.0.1:12345"}
)
```

### All config changing is admin only

The above API could exist as is without the concept of a `UserConfig`. Instead of only seeing and registering a partial provider configuration, a user could instead have access to all fields in the provider config (not the run config which has llama stack server settings). This would make the design of the API simpler but would necessitate authentication to be built into this since currently there is no concept of client accessible APIs that are only for certain users.

### The system is multi-tenant, and configurations are stored per-user on the server.

Rather than re-configuring all providers for all users, a `providers.register(...)` call could instead only change configuration for the calling user. This would require a pre-requisite of multi-tenancy, authentication, and a configuration registry similar to the models registry helpers.

This is a more complicated approach and perhaps one that can be an end-goal after initial implementation.

## Summary

The new Configuration API and expanded Inspect API allow a user to see parts of the stack configuration specifically, permitted parts of their provider configs. Additionally, a user can apply these corresponding fields in a "register" method that can update a stack configuration in place without needing to take down the server.
