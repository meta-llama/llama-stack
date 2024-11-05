# Fireworks Distribution

The `llamastack/distribution-fireworks` distribution consists of the following provider configurations.


| **API**         	| **Inference** 	| **Agents**     	| **Memory**                                       	| **Safety**     	| **Telemetry**  	|
|-----------------	|---------------	|----------------	|--------------------------------------------------	|----------------	|----------------	|
| **Provider(s)** 	| remote::fireworks   	| meta-reference 	| meta-reference 	| meta-reference 	| meta-reference 	|

### Step 0. Prerequisite
- Make sure you have access to a fireworks API Key. You can get one by visiting [fireworks.ai](https://fireworks.ai/)

### Step 1. Start the Distribution (Single Node CPU)

#### (Option 1) Start Distribution Via Docker
> [!NOTE]
> This assumes you have an hosted endpoint at Fireworks with API Key.

```
$ cd distributions/fireworks && docker compose up
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

#### (Option 2) Start Distribution Via Conda

```bash
llama stack build --template fireworks --image-type conda
# -- modify run.yaml to a valid Fireworks server endpoint
llama stack run ./run.yaml
```


### (Optional) Model Serving

Use `llama-stack-client models list` to check the available models served by Fireworks.
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
