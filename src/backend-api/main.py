import os 
from contextlib import asynccontextmanager

from fastapi import FastAPI

from db.qdrant.qdrant_client import QdrantStorage
from langchain_huggingface import HuggingFaceEmbeddings

from routes.query_routes import router as query_router
from routes.health_routes import router as health_router

from utils.logger import get_logger

logger = get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up the FastAPI application (OpenAI MVP Mode)...")

    # Initialize vector storage
    app.state.vectorstore = QdrantStorage()
    await app.state.vectorstore.initialize()

    # Initialize embedding model
    app.state.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load OpenRouter API key from environment variable
    # app.state.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", None)
    # if app.state.openrouter_api_key is None:
    #     logger.warning("OPENROUTER_API_KEY environment variable is not set. Make sure to provide it for OpenRouter access.")

    yield

    logger.info("Shutting down the FastAPI application...")    
    # Close vector storage connections
    await app.state.vectorstore.close()

app = FastAPI(
    title="Databricks expert RAG API", 
    version='1.0',
    description="API for an Databricks technical expert Retrieval Augmented Generation (RAG) expert",
    lifespan=lifespan
)

app.include_router(query_router, prefix="/query", tags=["query"])
app.include_router(health_router, tags=["health"])

# Middleware and exception handlers ??? (what this does??)

# Routers and other stuff with the API endpoints


if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=True,  # Enable auto-reload for development
    )

