# Fireworks Distribution

The `llamastack/distribution-` distribution consists of the following provider configurations.


| **API**         	| **Inference** 	| **Agents**     	| **Memory**                                       	| **Safety**     	| **Telemetry**  	|
|-----------------	|---------------	|----------------	|--------------------------------------------------	|----------------	|----------------	|
| **Provider(s)** 	| remote::fireworks   	| meta-reference 	| meta-reference 	| meta-reference 	| meta-reference 	|


### Docker: Start the Distribution (Single Node CPU)

> [!NOTE]
> This assumes you have an hosted endpoint at Fireworks with API Key.

```
$ cd distributions/fireworks
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

### Conda: llama stack run (Single Node CPU)

**Via Conda**

```bash
llama stack build --template fireworks --image-type conda
# -- modify run.yaml to a valid Fireworks server endpoint
llama stack run ./run.yaml
```


### Model Serving

Use `llama-stack-client models list` to chekc the available models served by Fireworks.
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
