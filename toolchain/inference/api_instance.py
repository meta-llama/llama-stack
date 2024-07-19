from .api.config import ImplType, ModelInferenceConfig


async def get_inference_api_instance(config: ModelInferenceConfig):
    if config.impl_config.impl_type == ImplType.inline.value:
        from .inference import ModelInferenceImpl

        return ModelInferenceImpl(config.impl_config)

    from .client import ModelInferenceClient

    return ModelInferenceClient(config.impl_config.url)
