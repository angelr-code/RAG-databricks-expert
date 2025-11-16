import os 
from contextlib import asynccontextmanager

from fastapi import FastAPI

from db.qdrant.qdrant_client import QdrantStorage
from langchain_huggingface import HuggingFaceEmbeddings

from utils.logger import get_logger

logger = get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up the FastAPI application...")

    # Initialize vector storage
    app.state.vectorstorage = QdrantStorage()
    await app.state.vectorstorage.initialize()

    # Initialize embedding model
    app.state.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load OpenRouter API key from environment variable
    app.state.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", None)
    if app.state.openrouter_api_key is None:
        logger.warning("OPENROUTER_API_KEY environment variable is not set. Make sure to provide it for OpenRouter access.")

    yield

    logger.info("Shutting down the FastAPI application...")    
    # Close vector storage connections
    await app.state.vectorstorage.close()

app = FastAPI(
    title="Databricks expert RAG API", 
    version='1.0',
    description="API for an Databricks technical expert Retrieval Augmented Generation (RAG) expert",
    lifespan=lifespan
)

# Middleware and exception handlers ??? (what this does??)

# Routers and other stuff with the API endpoints


if __name__ == '__main__':
    pass
    # The main method runs the API using uvicorn 

