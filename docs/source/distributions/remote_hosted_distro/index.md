---
orphan: true
---
# Remote-Hosted Distributions

Remote-Hosted distributions are available endpoints serving Llama Stack API that you can directly connect to.

| Distribution | Endpoint | Inference | Agents | Memory | Safety | Telemetry |
|-------------|----------|-----------|---------|---------|---------|------------|
| Together | [https://llama-stack.together.ai](https://llama-stack.together.ai) | remote::together | meta-reference | remote::weaviate | meta-reference | meta-reference |
| Fireworks | [https://llamastack-preview.fireworks.ai](https://llamastack-preview.fireworks.ai) | remote::fireworks | meta-reference | remote::weaviate | meta-reference | meta-reference |
| Nutanix | [https://llamastack-preview.nutanix.ai](https://llamastack-preview.nutanix.ai) | remote::nutanix | meta-reference | meta-reference  | meta-reference | meta-reference |

## Connecting to Remote-Hosted Distributions

You can use `llama-stack-client` to interact with these endpoints. For example, to list the available models served by the Fireworks endpoint:

```bash
$ pip install llama-stack-client
$ llama-stack-client configure --endpoint https://llamastack-preview.fireworks.ai
$ llama-stack-client models list
```

You will see outputs:
```
$ llama-stack-client models list
+------------------------------+------------------------------+---------------+------------+
| identifier                   | llama_model                  | provider_id   | metadata   |
+==============================+==============================+===============+============+
| Llama3.1-8B-Instruct         | Llama3.1-8B-Instruct         | fireworks0    | {}         |
+------------------------------+------------------------------+---------------+------------+
| Llama3.1-70B-Instruct        | Llama3.1-70B-Instruct        | fireworks0    | {}         |
+------------------------------+------------------------------+---------------+------------+
| Llama3.1-405B-Instruct       | Llama3.1-405B-Instruct       | fireworks0    | {}         |
+------------------------------+------------------------------+---------------+------------+
| Llama3.2-1B-Instruct         | Llama3.2-1B-Instruct         | fireworks0    | {}         |
+------------------------------+------------------------------+---------------+------------+
| Llama3.2-3B-Instruct         | Llama3.2-3B-Instruct         | fireworks0    | {}         |
+------------------------------+------------------------------+---------------+------------+
| Llama3.2-11B-Vision-Instruct | Llama3.2-11B-Vision-Instruct | fireworks0    | {}         |
+------------------------------+------------------------------+---------------+------------+
| Llama3.2-90B-Vision-Instruct | Llama3.2-90B-Vision-Instruct | fireworks0    | {}         |
+------------------------------+------------------------------+---------------+------------+
```

Checkout the [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python/blob/main/docs/cli_reference.md) repo for more details on how to use the `llama-stack-client` CLI. Checkout [llama-stack-app](https://github.com/meta-llama/llama-stack-apps/tree/main) for examples applications built on top of Llama Stack.
