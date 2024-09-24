import json
import uuid
from typing import List, Optional, Dict, Any
from numpy.typing import NDArray

import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import Auth

from llama_stack.apis.memory import *
from llama_stack.distribution.request_headers import get_request_provider_data
from llama_stack.providers.utils.memory.vector_store import (
    BankWithIndex,
    EmbeddingIndex,
)

from .config import WeaviateConfig, WeaviateRequestProviderData

class WeaviateIndex(EmbeddingIndex):
    def __init__(self, client: weaviate.Client, collection: str):
        self.client = client
        self.collection = collection

    async def add_chunks(self, chunks: List[Chunk], embeddings: NDArray):
        assert len(chunks) == len(embeddings), f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"

        data_objects = []
        for i, chunk in enumerate(chunks):
            print(f"Adding chunk #{i} tokens={chunk.token_count}")
            
            data_objects.append(wvc.data.DataObject(
                properties={
                    "chunk_content": chunk,
                },
                vector = embeddings[i].tolist()
            ))

        # Inserting chunks into a prespecified Weaviate collection
        assert self.collection is not None, "Collection name must be specified"
        my_collection = self.client.collections.get(self.collection)
        
        await my_collection.data.insert_many(data_objects)


    async def query(self, embedding: NDArray, k: int) -> QueryDocumentsResponse:
        assert self.collection is not None, "Collection name must be specified"

        my_collection = self.client.collections.get(self.collection)
        
        results = my_collection.query.near_vector(
            near_vector = embedding.tolist(),
            limit = k,
            return_meta_data = wvc.query.MetadataQuery(distance=True)
        )

        chunks = []
        scores = []
        for doc in results.objects:
            try:
                chunk = doc.properties["chunk_content"]
                chunks.append(chunk)
                scores.append(1.0 / doc.metadata.distance)
            
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Failed to parse document: {e}")

        return QueryDocumentsResponse(chunks=chunks, scores=scores)


class WeaviateMemoryAdapter(Memory):
    def __init__(self, config: WeaviateConfig) -> None:
        self.config = config
        self.client = None
        self.cache = {}

    def _get_client(self) -> weaviate.Client:
            request_provider_data = get_request_provider_data()
            
            if request_provider_data is not None:
                assert isinstance(request_provider_data, WeaviateRequestProviderData)

                print(f"WEAVIATE API KEY: {request_provider_data.weaviate_api_key}")
                print(f"WEAVIATE CLUSTER URL: {request_provider_data.weaviate_cluster_url}")
            
            # Connect to Weaviate Cloud
            return weaviate.connect_to_weaviate_cloud(
                cluster_url = request_provider_data.weaviate_cluster_url,
                auth_credentials = Auth.api_key(request_provider_data.weaviate_api_key),
                )

    async def initialize(self) -> None:
        try:
            self.client = self._get_client()

            # Create collection if it doesn't exist
            if not self.client.collections.exists(self.config.collection):
               self.client.collections.create(
                    name = self.config.collection,
                    vectorizer_config = wvc.config.Configure.Vectorizer.none(),
                    properties=[
                        wvc.config.Property(
                        name="chunk_content",
                        data_type=wvc.config.DataType.TEXT,
                        ),
                    ]
                )

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError("Could not connect to Weaviate server") from e

    async def shutdown(self) -> None:
        self.client = self._get_client()

        if self.client:
            self.client.close()

    async def create_memory_bank(
        self,
        name: str,
        config: MemoryBankConfig,
        url: Optional[URL] = None,
    ) -> MemoryBank:
        bank_id = str(uuid.uuid4())
        bank = MemoryBank(
            bank_id=bank_id,
            name=name,
            config=config,
            url=url,
        )
        self.client = self._get_client()
        
        # Store the bank as a new collection in Weaviate
        self.client.collections.create(
            name=bank_id
        )

        index = BankWithIndex(
            bank=bank,
            index=WeaviateIndex(cleint = self.client, collection = bank_id),
        )
        self.cache[bank_id] = index
        return bank

    async def get_memory_bank(self, bank_id: str) -> Optional[MemoryBank]:
        bank_index = await self._get_and_cache_bank_index(bank_id)
        if bank_index is None:
            return None
        return bank_index.bank

    async def _get_and_cache_bank_index(self, bank_id: str) -> Optional[BankWithIndex]:
        
        self.client = self._get_client()

        if bank_id in self.cache:
            return self.cache[bank_id]

        collections = await self.client.collections.list_all().keys()

        for collection in collections:
            if collection == bank_id:
                bank = MemoryBank(**json.loads(collection.metadata["bank"]))
                index = BankWithIndex(
                    bank=bank,
                    index=WeaviateIndex(self.client, collection),
                )
                self.cache[bank_id] = index
                return index

        return None

    async def insert_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
    ) -> None:
        index = await self._get_and_cache_bank_index(bank_id)
        if not index:
            raise ValueError(f"Bank {bank_id} not found")

        await index.insert_documents(documents)

    async def query_documents(
        self,
        bank_id: str,
        query: InterleavedTextMedia,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryDocumentsResponse:
        index = await self._get_and_cache_bank_index(bank_id)
        if not index:
            raise ValueError(f"Bank {bank_id} not found")

        return await index.query_documents(query, params)