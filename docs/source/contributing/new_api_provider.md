# Adding a New API Provider

This guide will walk you through the process of adding a new API provider to Llama Stack.


- Begin by reviewing the [core concepts](../concepts/index.md) of Llama Stack and choose the API your provider belongs to (Inference, Safety, VectorIO, etc.)
- Determine the provider type ({repopath}`Remote::llama_stack/providers/remote` or {repopath}`Inline::llama_stack/providers/inline`). Remote providers make requests to external services, while inline providers execute implementation locally.
- Add your provider to the appropriate {repopath}`Registry::llama_stack/providers/registry/`. Specify pip dependencies necessary.
- Update any distribution {repopath}`Templates::llama_stack/templates/` `build.yaml` and `run.yaml` files if they should include your provider by default. Run {repopath}`./scripts/distro_codegen.py` if necessary. Note that `distro_codegen.py` will fail if the new provider causes any distribution template to attempt to import provider-specific dependencies. This usually means the distribution's `get_distribution_template()` code path should only import any necessary Config or model alias definitions from each provider and not the provider's actual implementation.


Here are some example PRs to help you get started:
   - [Grok Inference Implementation](https://github.com/meta-llama/llama-stack/pull/609)
   - [Nvidia Inference Implementation](https://github.com/meta-llama/llama-stack/pull/355)
   - [Model context protocol Tool Runtime](https://github.com/meta-llama/llama-stack/pull/665)


## Testing the Provider

Before running tests, you must have required dependencies installed. This depends on the providers or distributions you are testing. For example, if you are testing the `together` distribution, you should install dependencies via `llama stack build --template together`.

### 1. Integration Testing

Integration tests are located in {repopath}`tests/integration`. These tests use the python client-SDK APIs (from the `llama_stack_client` package) to test functionality. Since these tests use client APIs, they can be run either by pointing to an instance of the Llama Stack server or "inline" by using `LlamaStackAsLibraryClient`.

Consult {repopath}`tests/integration/README.md` for more details on how to run the tests.

Note that each provider's `sample_run_config()` method (in the configuration class for that provider)
 typically references some environment variables for specifying API keys and the like. You can set these in the environment or pass these via the `--env` flag to the test command.


### 2. Unit Testing

Unit tests are located in {repopath}`tests/unit`. Provider-specific unit tests are located in {repopath}`tests/unit/providers`. These tests are all run automatically as part of the CI process.

Consult {repopath}`tests/unit/README.md` for more details on how to run the tests manually.

### 3. Additional end-to-end testing

1. Start a Llama Stack server with your new provider
2. Verify compatibility with existing client scripts in the [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main) repository
3. Document which scripts are compatible with your provider

## Submitting Your PR

1. Ensure all tests pass
2. Include a comprehensive test plan in your PR summary
3. Document any known limitations or considerations
