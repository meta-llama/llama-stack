# Developer Guide: Adding a New API Provider

This guide contains references to walk you through adding a new API provider.

### Adding a new API provider
1. First, decide which API your provider falls into (e.g. Inference, Safety, Agents, Memory).
2. Decide whether your provider is a remote provider, or inline implmentation. A remote provider is a provider that makes a remote request to an service. An inline provider is a provider where implementation is executed locally. Checkout the examples, and follow the structure to add your own API provider. Please find the following code pointers:

    - [Remote Adapters](https://github.com/meta-llama/llama-stack/tree/main/llama_stack/providers/remote)
    - [Inline Providers](https://github.com/meta-llama/llama-stack/tree/main/llama_stack/providers/inline)

3. [Build a Llama Stack distribution](https://llama-stack.readthedocs.io/en/latest/distribution_dev/building_distro.html) with your API provider.
4. Test your code!

### Testing your newly added API providers

1. Start with an _integration test_ for your provider. That means we will instantiate the real provider, pass it real configuration and if it is a remote service, we will actually hit the remote service. We **strongly** discourage mocking for these tests at the provider level. Llama Stack is first and foremost about integration so we need to make sure stuff works end-to-end. See [llama_stack/providers/tests/inference/test_inference.py](../llama_stack/providers/tests/inference/test_inference.py) for an example.

2. In addition, if you want to unit test functionality within your provider, feel free to do so. You can find some tests in `tests/` but they aren't well supported so far.

3. Test with a client-server Llama Stack setup. (a) Start a Llama Stack server with your own distribution which includes the new provider. (b) Send a client request to the server. See `llama_stack/apis/<api>/client.py` for how this is done. These client scripts can serve as lightweight tests.

You can find more complex client scripts [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main) repo. Note down which scripts works and do not work with your distribution.

### Submit your PR
After you have fully tested your newly added API provider, submit a PR with the attached test plan. You must have a Test Plan in the summary section of your PR.
