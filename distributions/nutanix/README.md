# Nutanix Distribution

The `llamastack/distribution-nutanix` distribution consists of the following provider configurations.


| **API**         	| **Inference** 	| **Agents**     	| **Memory**                                       	| **Safety**     	| **Telemetry**  	|
|-----------------	|---------------	|----------------	|--------------------------------------------------	|----------------	|----------------	|
| **Provider(s)** 	| remote::nutanix   	| meta-reference 	| meta-reference 	| meta-reference 	| meta-reference 	|


### Start the Distribution (Hosted remote)

> [!NOTE]
> This assumes you have an hosted Nutanix AI endpoint and an API Key.

1. Clone the repo
```
git clone git@github.com:meta-llama/llama-stack.git
cd llama-stack
```

2. Config the model name

Please adjust the `NUTANIX_SUPPORTED_MODELS` variable at line 29 in `llama_stack/providers/adapters/inference/nutanix/nutanix.py` according to your deployment.

3. Build the distrbution
```
pip install -e .
llama stack build --template nutanix --name ntnx --image-type conda
```

4. Set the endpoint URL and API Key
```
llama stack configure ntnx
```

5. Serve and enjoy!
```
llama stack run ntnx --port 174
```
