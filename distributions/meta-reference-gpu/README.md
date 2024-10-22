# Meta Reference Distribution

The `llamastack/distribution-meta-reference-gpu` distribution consists of the following provider configurations.


| **API**         	| **Inference** 	| **Agents**     	| **Memory**                                       	| **Safety**     	| **Telemetry**  	|
|-----------------	|---------------	|----------------	|--------------------------------------------------	|----------------	|----------------	|
| **Provider(s)** 	| meta-reference  	| meta-reference 	| meta-reference, remote::pgvector, remote::chroma 	| meta-reference 	| meta-reference 	|


### Start the Distribution (Single Node GPU)

> [!NOTE]
> This assumes you have access to GPU to start a local server with access to your GPU.


> [!NOTE]
> `~/.llama` should be the path containing downloaded weights of Llama models.


To download and start running a pre-built docker container, you may use the following commands:

```
docker run -it -p 5000:5000 -v ~/.llama:/root/.llama -v ./run.yaml:/root/my-run.yaml --gpus=all distribution-meta-reference-gpu --yaml_config /root/my-run.yaml
```

### Alternative (Build and start distribution locally via conda)
- You may checkout the [Getting Started](../../docs/getting_started.md) for more details on building locally via conda and starting up a meta-reference distribution.
