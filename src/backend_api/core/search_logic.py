import opik

from src.backend_api.models.api_models import SearchResult
from src.db.qdrant.qdrant_client import QdrantStorage

from src.utils.logger import setup_logging

logger = setup_logging()

@opik.track(name="rag_retrieval")
async def search_service(query_text: str, vectorstore: QdrantStorage, limit: int = 5) -> SearchResult:
    """
    Performs a search in the Vector DB based on the query_text and optional keywords.

    Args:
        query_text (str): The user input text to search for.
        vectorstore (QdrantStorage): The vector storage instance to perform the search on.
        embedding_model (HuggingFaceEmbeddings): The embedding model to convert text to vectors
        keywords (str | None): Optional keywords to filter the search results.
        limit (int): The maximum number of chunks to retrieve.
    Returns:
        SearchResult: An object containing the retrieved contexts and their associated sources.
    """
    logger.info(f"Received search request with query_text: '{query_text}', limit: {limit}")

    results = await vectorstore.hybrid_search(
        query_text=query_text,
        top_k=limit
    )

    logger.info(f"Retrieval phase completed successfully. Found {len(results.contexts)} matching contexts.")
    return results