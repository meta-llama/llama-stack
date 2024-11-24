from pydantic import BaseModel

from .config import SambanovaImplConfig

class SambanovaProviderDataValidator(BaseModel):
    sambanova_api_key: str


async def get_adapter_impl(config: SambanovaImplConfig, _deps):
    from .sambanova import SambanovaInferenceAdapter

    assert isinstance(
        config, SambanovaImplConfig
    ), f"Unexpected config type: {type(config)}"
    impl = SambanovaInferenceAdapter(config)
    await impl.initialize()
    return impl
