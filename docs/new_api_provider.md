# Developer Guide: Adding a New API Provider

This guide contains references to walk you through

### Adding a new API provider
1. First, decide which API your provider falls into (e.g. Inference, Safety, Agents, Memory).
2. Decide whether your provider is a remote provider, or inline implmentation. A remote provider is a provider that makes a remote request to an service. An inline provider is a provider where implementation is executed locally. Checkout the examples, and follow the structure to add your own API provider. Please find the following code pointers:
  - [Inference Remote Adapter](../llama_stack/providers/adapters/inference/)
  - [Inference Inline Provider](../llama_stack/providers/impls/)
3. [Build a Llama Stack distribution](./building_distro.md) with your API provider.
4. Test your code!

### Testing your newly added API providers
1. Start Llama Stack server with your
2. Test with sending a client request to the server.
3. Add tests for your newly added provider. See [tests/](../tests/) for example unit tests.
4. Test the supported functionalities for your provider using our providers tests infra. See [llama_stack/providers/tests/<api>/test_<api>](../llama_stack/providers/tests/inference/test_inference.py).

### Submit your PR
After you have fully tested your newly added API provider, submit a PR with the attached test plan, and we will help you verify the necessary requirements.
