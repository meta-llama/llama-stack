# Together Distribution

### Connect to a Llama Stack Together Endpoint
- You may connect to a hosted endpoint `https://llama-stack.together.ai`, serving a Llama Stack distribution

The `llamastack/distribution-together` distribution consists of the following provider configurations.


| **API**         	| **Inference** 	| **Agents**     	| **Memory**                                       	| **Safety**     	| **Telemetry**  	|
|-----------------	|---------------	|----------------	|--------------------------------------------------	|----------------	|----------------	|
| **Provider(s)** 	| remote::together   	| meta-reference 	| meta-reference, remote::weaviate 	| meta-reference 	| meta-reference 	|


### Start the Distribution (Single Node CPU)

> [!NOTE]
> This assumes you have an hosted endpoint at Together with API Key.

```
$ cd distributions/together
$ ls
compose.yaml  run.yaml
$ docker compose up
```

Make sure in you `run.yaml` file, you inference provider is pointing to the correct Together URL server endpoint. E.g.
```
inference:
  - provider_id: together
    provider_type: remote::together
    config:
      url: https://api.together.xyz/v1
      api_key: <optional api key>
```

### (Alternative) llama stack run (Single Node CPU)

```
docker run --network host -it -p 5000:5000 -v ./run.yaml:/root/my-run.yaml --gpus=all llamastack/distribution-together --yaml_config /root/my-run.yaml
```

Make sure in you `run.yaml` file, you inference provider is pointing to the correct Together URL server endpoint. E.g.
```
inference:
  - provider_id: together
    provider_type: remote::together
    config:
      url: https://api.together.xyz/v1
      api_key: <optional api key>
```

**Via Conda**

```bash
llama stack build --template together --image-type conda
# -- modify run.yaml to a valid Together server endpoint
llama stack run ./run.yaml
```
