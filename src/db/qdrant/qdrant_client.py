from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
import os

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 384))


class QdrantStorage():
    def __init__(self, url=QDRANT_URL, collection=QDRANT_COLLECTION, dim=EMBEDDING_DIM):
        self.client = AsyncQdrantClient(url, timeout=15)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    async def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id = ids[i], vector = vectors[i], payload = payloads[i]) for i in range(len(ids))]
        try:
            self.client.upsert(self.collection, points = points)
            print(f"Ingestion of {len(ids)} vectors to Qdrant completed")
        except Exception as e:
            print(f"Vector ingestion failed: {e}")

    async def search(self, query_vector, top_k = 5):
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
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
            self.client.delete(
                self.collection,
                points_selector=filter
            )
            print(f"Deleted all chunks for document {document_id}: {e}")
            return True
        except Exception as e:
            print(f"Error deleting chunks for document {document_id}: {e}")
            return False
        
    async def close(self):
        self.client.close()
