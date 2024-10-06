from typing import Any

from .config import VLLMConfig


async def get_provider_impl(config: VLLMConfig, _deps) -> Any:
    from .vllm import VLLMInferenceImpl

    impl = VLLMInferenceImpl(config)
    await impl.initialize()
    return impl
