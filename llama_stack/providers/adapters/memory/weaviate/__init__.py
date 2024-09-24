from .config import WeaviateConfig

async def get_adapter_impl(config: WeaviateConfig, _deps):
    from .weaviate import WeaviateMemoryAdapter

    impl = WeaviateMemoryAdapter(config)
    await impl.initialize()
    return impl