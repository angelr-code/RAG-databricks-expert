import os
import asyncio
import uuid

from typing import List, Dict, Any

from fastembed import TextEmbedding, SparseTextEmbedding

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    VectorParams, 
    Distance, 
    PointStruct, 
    Filter, 
    FieldCondition, 
    MatchValue,
    KeywordIndexParams,
    KeywordIndexType,
    SparseVector,
    SparseVectorParams,
    SparseIndexParams,
    Prefetch,
    FusionQuery,
    Fusion
)

from src.utils.logger import setup_logging

from src.backend_api.models.api_models import SearchResult

logger = setup_logging()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 384))


class QdrantStorage():
    def __init__(self, url=QDRANT_URL, collection=QDRANT_COLLECTION, dim=EMBEDDING_DIM):
        self.client = AsyncQdrantClient(url, api_key=QDRANT_API_KEY,timeout=15)
        self.collection = collection
        self.dim = dim
        self.dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    async def initialize(self):
        """
        Initializes the Qdrant collection in an asynchronous manner with hybrid vector configuration.
        Creates the collection if it does not exist and sets up a doc_id payload index to make operations
        like deletes faster.
        """
        try:
            exists = await self.client.collection_exists(self.collection)

            if not exists:
                await self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config={
                        "dense_vector": VectorParams(
                            size=self.dim, 
                            distance=Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse_vector": SparseVectorParams(
                            index=SparseIndexParams(
                                on_disk=False
                            )
                        )
                    }
                )

                await self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name="document_id", 
                    field_schema=KeywordIndexParams(
                        type=KeywordIndexType.KEYWORD
                    ),
                    wait=True
                )
                logger.info(f"Collection '{self.collection}' created successfully in Qdrant with Hybrid Vector Configuration")

            else: 
                logger.info("Qdrant collection already exists")

        except Exception as e:
            logger.error(f"Error initializing Qdrant: {e}")
            raise

    async def upsert(self, chunks: List[str], metadatas: List[Dict[str, Any]]):
        """
        Upserts the given chunks and their associated metadata into the Qdrant collection.

        Args:
            chunks (List[str]): The list of text chunks to be upserted.
            metadatas (List[Dict[str, Any]]): The list of metadata dictionaries associated with each chunk.
        """
        try:
            dense_vectors = await asyncio.to_thread(lambda: list(self.dense_model.embed(chunks)))
            sparse_vectors = await asyncio.to_thread(lambda: list(self.sparse_model.embed(chunks)))
            
            points = [PointStruct(id = str(uuid.uuid4()), 
                                  vector = {
                                      "dense_vector": dense_vectors[i].tolist(),
                                      "sparse_vector": SparseVector(
                                          indices=sparse_vectors[i].indices.tolist(),
                                          values=sparse_vectors[i].values.tolist()
                                      )
                                  }, 
                                  payload = {
                                      "text": chunks[i],
                                      **metadatas[i]
                                  }
                                ) 
                    for i in range(len(chunks))]

            await self.client.upsert(collection_name=self.collection, points = points)

            logger.info(f"Ingestion of {len(chunks)} vectors to Qdrant completed")
        except Exception as e:
            logger.error(f"Vector ingestion failed: {e}")
            raise

    async def hybrid_search(self, query_text: str, top_k: int = 5) -> SearchResult:
        """
        Performs a hybrid search (dense + sparse) in the Qdrant collection based on the query_text.
        Uses RRF (Reciprocal Rank Fusion) to combine results from both searches retrieving the most 
        relevant ones in both semantic and keyword aspects.

        Args:
            query_text (str): The user input text to search for.
            top_k (int): The maximum number of chunks to retrieve.
        Returns:
            Dict[str, Any]: A dictionary containing the retrieved contexts and their associated sources.
        """
        query_dense = list(self.dense_model.embed([query_text]))[0]
        query_sparse = list(self.sparse_model.embed([query_text]))[0]

        qdrant_sparse_vector = SparseVector(    
            indices=query_sparse.indices,
            values=query_sparse.values
        )

        # We perform hybrid search for the result to have both conceptual meaning and the adequate technical terms
        results = await self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(
                    query=list(query_dense),
                    using="dense_vector",
                    limit=top_k * 2
                ),
                Prefetch(
                    query=qdrant_sparse_vector,
                    using="sparse_vector",
                    limit=top_k * 2
                )
            ],  # We perform two searches (hybrid search): dense (semantic) and sparse (keyword)
            query=FusionQuery(fusion=Fusion.RRF), # Reciprocal Rank Fusion is the algorithm used to combine results. 
            with_payload=True,
            limit=top_k
        )

        contexts = []
        sources = set() # set in order to not repeat sources
        for result in results.points:
            payload = result.payload
            text = payload.get("text", "")
            source = payload.get("source_url", "")
            if source:
                contexts.append(text)
                sources.add(source)

        return SearchResult(
            contexts=contexts,
            sources=list(sources)
        )
    
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
    