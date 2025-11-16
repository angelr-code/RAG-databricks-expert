import opik
from fastapi import Request

from loguru import logger

from models.api_models import SearchResult

from db.qdrant.qdrant_client import QdrantStorage

@opik.track
async def query(request: Request,query_text: str = "",keywords: str | None = None, limit: int = 5) -> SearchResult:
    logger.info(f"Received search request with query_text: '{query_text}', keywords: '{keywords}', limit: {limit}")
    vectorstore: QdrantStorage = request.app.state.vectorstore
    embedding_model = request.app.state.embedding_model

    query_vector = embedding_model.encode_text(query_text)
    search_results = await vectorstore.search(query_vector, keywords, top_k=limit)

    logger.info(f"Search completed. Returning {len(search_results['contexts'])} associated contexts.")
    return search_results