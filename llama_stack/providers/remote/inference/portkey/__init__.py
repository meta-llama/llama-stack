
from .config import PortkeyImplConfig


async def get_adapter_impl(config: PortkeyImplConfig, _deps):
    from .portkey import PortkeyInferenceAdapter

    assert isinstance(
        config, PortkeyImplConfig
    ), f"Unexpected config type: {type(config)}"

    impl = PortkeyInferenceAdapter(config)

    await impl.initialize()

    return impl
