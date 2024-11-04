# Hosted Distribution

Hosted distributions are distributions connecting to remote hosted services through Llama Stack server. Inference is done through remote providers. These are useful if you have an API key for a remote inference provider like Fireworks, Together, etc.

| **Distribution** 	|           **Llama Stack Docker**           	| Start This Distribution 	|    **Inference**   	|     **Agents**     	|     **Memory**     	|     **Safety**     	|    **Telemetry**   	|
|:----------------:	|:------------------------------------------:	|:-----------------------:	|:------------------:	|:------------------:	|:------------------:	|:------------------:	|:------------------:	|
|        Together       	|         [llamastack/distribution-together](https://hub.docker.com/repository/docker/llamastack/distribution-together/general)        	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/hosted_distro/together.html)       	| remote::together 	| meta-reference | remote::weaviate | meta-reference 	| meta-reference  	|
|        Fireworks       	|         [llamastack/distribution-fireworks](https://hub.docker.com/repository/docker/llamastack/distribution-fireworks/general)        	|       [Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/distributions/hosted_distro/fireworks.html)       	| remote::fireworks 	| meta-reference | remote::weaviate | meta-reference 	| meta-reference  	|

```{toctree}
:maxdepth: 1

fireworks
together
```
