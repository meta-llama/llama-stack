# Meta Reference Distribution

The `llamastack/distribution-meta-reference-gpu` distribution consists of the following provider configurations.


| **API**         	| **Inference** 	| **Agents**     	| **Memory**                                       	| **Safety**     	| **Telemetry**  	|
|-----------------	|---------------	|----------------	|--------------------------------------------------	|----------------	|----------------	|
| **Provider(s)** 	| meta-reference  	| meta-reference 	| meta-reference, remote::pgvector, remote::chroma 	| meta-reference 	| meta-reference 	|


### Start the Distribution (Single Node GPU)

> [!NOTE]
> This assumes you have access to GPU to start a TGI server with access to your GPU.

> [!NOTE]
> For GPU inference, you need to set these environment variables for specifying local directory containing your model checkpoints, and enable GPU inference to start running docker container.
```
export LLAMA_CHECKPOINT_DIR=~/.llama
```

> [!NOTE]
> `~/.llama` should be the path containing downloaded weights of Llama models.


To download and start running a pre-built docker container, you may use the following commands:

```
docker run -it -p 5000:5000 -v ~/.llama:/root/.llama --gpus=all llamastack/llamastack-local-gpu
```

### Alternative (Build and start distribution locally via conda)
- You may checkout the [Getting Started](../../docs/getting_started.md) for more details on starting up a meta-reference distribution.
