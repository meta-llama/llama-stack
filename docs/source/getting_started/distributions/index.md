# Llama Stack Distribution

A Distribution is where APIs and Providers are assembled together to provide a consistent whole to the end application developer. You can mix-and-match providers -- some could be backed by local code and some could be remote. As a hobbyist, you can serve a small model locally, but can choose a cloud provider for a large model. Regardless, the higher level APIs your app needs to work with don't need to change at all. You can even imagine moving across the server / mobile-device boundary as well always using the same uniform set of APIs for developing Generative AI applications.

| **Distribution** 	|           **Llama Stack Docker**           	| Start This Distribution 	|    **Inference**   	|     **Agents**     	|     **Memory**     	|     **Safety**     	|    **Telemetry**   	|
|:----------------:	|:------------------------------------------:	|:-----------------------:	|:------------------:	|:------------------:	|:------------------:	|:------------------:	|:------------------:	|
|  Meta Reference  	| [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general) 	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/meta-reference-gpu.html)       	| meta-reference 	| meta-reference 	| meta-reference; remote::pgvector; remote::chromadb	| meta-reference 	| meta-reference	|
|  Meta Reference Quantized  	| [llamastack/distribution-meta-reference-quantized-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-quantized-gpu/general) 	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/meta-reference-quantized-gpu.html)       	| meta-reference-quantized 	| meta-reference 	| meta-reference; remote::pgvector; remote::chromadb	| meta-reference 	| meta-reference	|
|      Ollama      	|       [llamastack/distribution-ollama](https://hub.docker.com/repository/docker/llamastack/distribution-ollama/general)       	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/ollama.html)       	| remote::ollama	| meta-reference 	| remote::pgvector; remote::chromadb 	|  meta-reference 	| meta-reference 	|
|        TGI       	|         [llamastack/distribution-tgi](https://hub.docker.com/repository/docker/llamastack/distribution-tgi/general)        	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/tgi.html)       	| remote::tgi	| meta-reference 	| meta-reference; remote::pgvector; remote::chromadb 	| meta-reference 	| meta-reference 	|
|        Together       	|         [llamastack/distribution-together](https://hub.docker.com/repository/docker/llamastack/distribution-together/general)        	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/together.html)       	| remote::together 	| meta-reference | remote::weaviate | meta-reference 	| meta-reference  	|
|        Fireworks       	|         [llamastack/distribution-fireworks](https://hub.docker.com/repository/docker/llamastack/distribution-fireworks/general)        	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/fireworks.html)       	| remote::fireworks 	| meta-reference | remote::weaviate | meta-reference 	| meta-reference  	|

```{toctree}
:maxdepth: 1

meta-reference-gpu
meta-reference-quantized-gpu
ollama
tgi
together
fireworks
dell-tgi
```
