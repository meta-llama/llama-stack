# Fireworks Distribution

The `llamastack/distribution-` distribution consists of the following provider configurations.


| **API**         	| **Inference** 	| **Agents**     	| **Memory**                                       	| **Safety**     	| **Telemetry**  	|
|-----------------	|---------------	|----------------	|--------------------------------------------------	|----------------	|----------------	|
| **Provider(s)** 	| remote::fireworks   	| meta-reference 	| meta-reference 	| meta-reference 	| meta-reference 	|


### Start the Distribution (Single Node CPU)

> [!NOTE]
> This assumes you have an hosted endpoint at Fireworks with API Key.

```
$ cd llama-stack/distribution/fireworks
$ ls
compose.yaml  run.yaml
$ docker compose up
```

Make sure in you `run.yaml` file, you inference provider is pointing to the correct Fireworks URL server endpoint. E.g.
```
inference:
  - provider_id: fireworks
    provider_type: remote::fireworks
    config:
      url: https://api.fireworks.ai/inferenc
      api_key: <optional api key>
```

### (Alternative) TGI server + llama stack run (Single Node GPU)

```
docker run --network host -it -p 5000:5000 -v ./run.yaml:/root/my-run.yaml --gpus=all llamastack/distribution-fireworks --yaml_config /root/my-run.yaml
```

Make sure in you `run.yaml` file, you inference provider is pointing to the correct Fireworks URL server endpoint. E.g.
```
inference:
  - provider_id: fireworks
    provider_type: remote::fireworks
    config:
      url: https://api.fireworks.ai/inference
      api_key: <optional api key>
```

**Via Conda**

```bash
llama stack build --config ./build.yaml
# -- modify run.yaml to a valid Fireworks server endpoint
llama stack run ./run.yaml
```
