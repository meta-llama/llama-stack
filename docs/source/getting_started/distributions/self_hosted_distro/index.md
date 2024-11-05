# Self-Hosted Distribution

We offer deployable distributions where you can host your own Llama Stack server using local inference.

| **Distribution** 	|           **Llama Stack Docker**           	| Start This Distribution 	|    **Inference**   	|     **Agents**     	|     **Memory**     	|     **Safety**     	|    **Telemetry**   	|
|:----------------:	|:------------------------------------------:	|:-----------------------:	|:------------------:	|:------------------:	|:------------------:	|:------------------:	|:------------------:	|
|  Meta Reference  	| [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general) 	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/meta-reference-gpu.html)       	| meta-reference 	| meta-reference 	| meta-reference; remote::pgvector; remote::chromadb	| meta-reference 	| meta-reference	|
|  Meta Reference Quantized  	| [llamastack/distribution-meta-reference-quantized-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-quantized-gpu/general) 	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/meta-reference-quantized-gpu.html)       	| meta-reference-quantized 	| meta-reference 	| meta-reference; remote::pgvector; remote::chromadb	| meta-reference 	| meta-reference	|
|      Ollama      	|       [llamastack/distribution-ollama](https://hub.docker.com/repository/docker/llamastack/distribution-ollama/general)       	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/ollama.html)       	| remote::ollama	| meta-reference 	| remote::pgvector; remote::chromadb 	|  meta-reference 	| meta-reference 	|
|        TGI       	|         [llamastack/distribution-tgi](https://hub.docker.com/repository/docker/llamastack/distribution-tgi/general)        	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/self_hosted_distro/tgi.html)       	| remote::tgi	| meta-reference 	| meta-reference; remote::pgvector; remote::chromadb 	| meta-reference 	| meta-reference 	|

```{toctree}
:maxdepth: 1

meta-reference-gpu
meta-reference-quantized-gpu
ollama
tgi
dell-tgi
```
