import os

from typing import List, Dict, Any, Optional

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

from src.utils.logger import setup_logging

logger = setup_logging()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 384))


class QdrantStorage():
    def __init__(self, url=QDRANT_URL, collection=QDRANT_COLLECTION, dim=EMBEDDING_DIM):
        self.client = AsyncQdrantClient(url, timeout=15)
        self.collection = collection
        self.dim = dim

    async def create_payload_index(self, field_name: str, field_type: str = "text"):
        """
        Creates a payload index on the specified field.

        Args:
            field_name (str): The name of the field to index.
            field_type (str): The type of the field.
        """
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
                collection_name=self.collection,
                field_name=field_name,
                field_schema=schema_params,
                wait=True 
            )
            logger.info(f"Payload index created successfully for: {field_name}")

        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Index for '{field_name}' already exists. Pass.")
            else:
                logger.error(f"Error creating payload index for field '{field_name}': {e}")

    async def initialize(self):
        """
        Initializes the Qdrant collection in an asynchronous manner.
        Creates the collection if it does not exist and sets up necessary payload indexes.
        """
        try:
            exists = await self.client.collection_exists(self.collection)

            if not exists:
                await self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
                )

                await self.create_payload_index(field_name="title", field_type="text")
                logger.info(f"Collection '{self.collection}' created successfully in Qdrant")

            else: 
                logger.info("Qdrant collection already exists")

        except Exception as e:
            logger.error(f"Error initializing Qdrant: {e}")
            raise

    async def upsert(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        """
        Upserts vectors and their associated payloads into the Qdrant collection.
        
        Args:
            ids (List[str]): List of unique identifiers for the vectors.
            vectors (List[List[float]]): List of vectors to be upserted.
            payloads (List[Dict[str, Any]]): List of payloads associated with each vector.
        """
        points = [PointStruct(id = ids[i], vector = vectors[i], payload = payloads[i]) for i in range(len(ids))]
        try:
            await self.client.upsert(self.collection, points = points)
            logger.info(f"Ingestion of {len(ids)} vectors to Qdrant completed")
        except Exception as e:
            logger.error(f"Vector ingestion failed: {e}")

    async def search(self, query_vector: List[float], keywords: Optional[str], top_k: int = 5) -> Dict[str, Any]:
        """
        Searches for the most similar vectors in the Qdrant collection based on the query vector and optional keywords.
        
        Args:
            query_vector (List[float]): The vector to search for similar vectors.
            keywords (Optional[str]): Optional keywords to filter the search results.
            top_k (int): The number of top similar vectors to retrieve.
        
        Returns:
            Dict[str, Any]: A dictionary containing the contexts and sources of the search results.
        """
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
    
    async def delete_by_document_id(self, document_id: str) -> bool:
        """
        Deletes al vectors associated with a given document_id
        
        Args:
            document_id (str): The ID of the document whose vectors are to be deleted.
            
        Returns:
            bool: True if deletion was successful, False otherwise.
        """
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
            logger.info(f"Deleted all chunks for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting chunks for document {document_id}: {e}")
            return False
        
        
    async def close(self):
        """Closes the Qdrant client connection."""
        await self.client.close()
        logger.info("Qdrant client connection closed.")
    