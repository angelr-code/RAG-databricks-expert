import os 
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.db.qdrant.qdrant_client import QdrantStorage

from src.backend_api.routes.query_routes import router as query_router
from src.backend_api.routes.health_routes import router as health_router

from src.utils.logger import setup_logging

logger = setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up the FastAPI application (OpenAI MVP Mode)...")

    # Initialize vector storage
    app.state.vectorstore = QdrantStorage()
    await app.state.vectorstore.initialize()

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

# Middleware and exception handlers ???

if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "src.backend_api.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=True  # auto-reload for development
    )

