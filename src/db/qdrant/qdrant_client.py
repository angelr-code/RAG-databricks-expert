import os

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    VectorParams, 
    Distance, 
    PointStruct, 
    Filter, 
    FieldCondition, 
    MatchValue,
    TextIndexParams, 
    TextIndexType,
    KeywordIndexParams,
    KeywordIndexType,
    TokenizerType,
    MatchText
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 384))


class QdrantStorage():
    def __init__(self, url=QDRANT_URL, collection=QDRANT_COLLECTION, dim=EMBEDDING_DIM):
        self.client = AsyncQdrantClient(url, timeout=15)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.create_payload_index(field_name="title", field_type="text")
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    async def initialize(self):
        try:
            exists = await self.client.collection_exists(self.collection_name)

            if not exists:
                await self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
                )

                await self.create_payload_index(field_name="title", field_type="text")
            else: 
                print("Collection already exists")

        except Exception as e:
            print(f"Error initializing Qdrant: {e}")
            raise

    async def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id = ids[i], vector = vectors[i], payload = payloads[i]) for i in range(len(ids))]
        try:
            await self.client.upsert(self.collection, points = points)
            print(f"Ingestion of {len(ids)} vectors to Qdrant completed")
        except Exception as e:
            print(f"Vector ingestion failed: {e}")

    async def search(self, query_vector, keywords, top_k = 5):
        query_filter = None
        if keywords:
            query_filter = Filter(
                must=[
                    FieldCondition(
                    key="title",
                    match=MatchText(text=keywords)
                    )
                ]
            )

        results = await self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            query_filter=query_filter,
            with_payload=True,
            limit=top_k
        )
        contexts = []
        sources = set() # set in order to not repeat sources

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source_url", "")
            if source:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}
    
    async def delete_by_document_id(self, document_id):
        """Deletes al vectors associated with a given document_id"""
        try:
            filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=str(document_id))
                    )
                ]
            )
            await self.client.delete(
                self.collection,
                points_selector=filter
            )
            print(f"Deleted all chunks for document {document_id}: {e}")
            return True
        except Exception as e:
            print(f"Error deleting chunks for document {document_id}: {e}")
            return False
        
    async def create_payload_index(self, field_name: str, field_type: str = "text"):
        try:
            if field_type == "text":
                # This will be the default use with the field 'title'
                schema_params = TextIndexParams(
                    type=TextIndexType.TEXT,
                    tokenizer=TokenizerType.WORD, # Split words palabras
                    min_token_len=3,              # Ignores short words
                    lowercase=True           
                )
            else:
                # If it's not text (e.g. int) -> exact match
                schema_params = KeywordIndexParams(type=KeywordIndexType.KEYWORD)

            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=schema_params,
                wait=True 
            )
            print(f"Payload index created successfully for: {field_name}")

        except Exception as e:
            if "already exists" in str(e):
                print(f"Index for '{field_name}' already exists. Pass.")
            else:
                print(f"Error creating payload index for field '{field_name}': {e}")
        
    async def close(self):
        await self.client.close()
