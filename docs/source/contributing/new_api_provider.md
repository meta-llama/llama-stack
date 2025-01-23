# Adding a New API Provider

This guide will walk you through the process of adding a new API provider to Llama Stack.

## Getting Started

1. **Choose Your API Category**
   - Determine which API category your provider belongs to (Inference, Safety, Agents, VectorIO)
   - Review the core concepts of Llama Stack in the [concepts guide](../concepts/index.md)

2. **Determine Provider Type**
   - **Remote Provider**: Makes requests to external services
   - **Inline Provider**: Executes implementation locally

   Reference existing implementations:
   - {repopath}`Remote Providers::llama_stack/providers/remote`
   - {repopath}`Inline Providers::llama_stack/providers/inline`

   Example PRs:
   - [Grok Inference Implementation](https://github.com/meta-llama/llama-stack/pull/609)
   - [Nvidia Inference Implementation](https://github.com/meta-llama/llama-stack/pull/355)
   - [Model context protocol Tool Runtime](https://github.com/meta-llama/llama-stack/pull/665)

3. **Register Your Provider**
   - Add your provider to the appropriate {repopath}`Registry::llama_stack/providers/registry/`
   - Specify any required pip dependencies

4. **Integration**
   - Update the run.yaml file to include your provider
   - To make your provider a default option or create a new distribution, look at the teamplates in {repopath}`llama_stack/templates/` and run {repopath}`llama_stack/scripts/distro_codegen.py`
   - Example PRs:
     - [Adding Model Context Protocol Tool Runtime](https://github.com/meta-llama/llama-stack/pull/816)

## Testing Guidelines

### 1. Integration Testing
- Create integration tests that use real provider instances and configurations
- For remote services, test actual API interactions
- Avoid mocking at the provider level
- Reference examples in {repopath}`tests/client-sdk`

### 2. Unit Testing (Optional)
- Add unit tests for provider-specific functionality
- See examples in {repopath}`llama_stack/providers/tests/inference/test_text_inference.py`

### 3. End-to-End Testing
1. Start a Llama Stack server with your new provider
2. Test using client requests
3. Verify compatibility with existing client scripts in the [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main) repository
4. Document which scripts are compatible with your provider

## Submitting Your PR

1. Ensure all tests pass
2. Include a comprehensive test plan in your PR summary
3. Document any known limitations or considerations
4. Submit your pull request for review
