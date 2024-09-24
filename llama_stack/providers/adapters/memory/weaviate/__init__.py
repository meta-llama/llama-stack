from llama_stack.distribution.datatypes import RemoteProviderConfig

async def get_adapter_impl(config: RemoteProviderConfig, _deps):
    from .weaviate import WeaviateMemoryAdapter

    impl = WeaviateMemoryAdapter(config.url, config.username, config.password)
    await impl.initialize()
    return impl