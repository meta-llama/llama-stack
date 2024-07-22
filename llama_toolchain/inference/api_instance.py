from .api.config import ImplType, InferenceConfig


async def get_inference_api_instance(config: InferenceConfig):
    if config.impl_config.impl_type == ImplType.inline.value:
        from .inference import InferenceImpl

        return InferenceImpl(config.impl_config)

    from .client import InferenceClient

    return InferenceClient(config.impl_config.url)
