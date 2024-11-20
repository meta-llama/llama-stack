# Bedrock Distribution
```{toctree}
:maxdepth: 2
:hidden:

self
```

### Connect to a Llama Stack Bedrock Endpoint
- You may connect to Amazon Bedrock APIs for running LLM inference

The `llamastack/distribution-bedrock` distribution consists of the following provider configurations.


| **API**         	| **Inference** 	| **Agents**     	| **Memory**     	| **Safety**     	| **Telemetry**  	|
|-----------------	|---------------	|----------------	|----------------	|----------------	|----------------	|
| **Provider(s)** 	| remote::bedrock | meta-reference 	| meta-reference 	| remote::bedrock | meta-reference 	|


### Docker: Start the Distribution (Single Node CPU)

> [!NOTE]
> This assumes you have valid AWS credentials configured with access to Amazon Bedrock.

```
$ cd distributions/bedrock && docker compose up
```

Make sure in your `run.yaml` file, your inference provider is pointing to the correct AWS configuration. E.g.
```
inference:
  - provider_id: bedrock0
    provider_type: remote::bedrock
    config:
      aws_access_key_id: <AWS_ACCESS_KEY_ID>
      aws_secret_access_key: <AWS_SECRET_ACCESS_KEY>
      aws_session_token: <AWS_SESSION_TOKEN>
      region_name: <AWS_REGION>
```

### Conda llama stack run (Single Node CPU)

```bash
llama stack build --template bedrock --image-type conda
# -- modify run.yaml with valid AWS credentials
llama stack run ./run.yaml
```

### (Optional) Update Model Serving Configuration

Use `llama-stack-client models list` to check the available models served by Amazon Bedrock.

```
$ llama-stack-client models list
+------------------------------+------------------------------+---------------+------------+
| identifier                   | llama_model                  | provider_id   | metadata   |
+==============================+==============================+===============+============+
| Llama3.1-8B-Instruct         | meta.llama3-1-8b-instruct-v1:0 | bedrock0     | {}         |
+------------------------------+------------------------------+---------------+------------+
| Llama3.1-70B-Instruct        | meta.llama3-1-70b-instruct-v1:0 | bedrock0     | {}         |
+------------------------------+------------------------------+---------------+------------+
| Llama3.1-405B-Instruct       | meta.llama3-1-405b-instruct-v1:0 | bedrock0     | {}         |
+------------------------------+------------------------------+---------------+------------+
```
