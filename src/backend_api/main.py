import os 
from time import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Security, HTTPException, Depends
from fastapi.security import APIKeyHeader
from mangum import Mangum

from src.db.qdrant.qdrant_client import QdrantStorage

from src.backend_api.routes.query_routes import router as query_router
from src.backend_api.routes.health_routes import router as health_router

from src.utils.logger import setup_logging

from dotenv import load_dotenv

logger = setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
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

# Middleware 
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware to log request processing time and details.
    
    Args:
        request (Request): The incoming HTTP request.
        call_next: The next middleware or route handler to call.
    
    Returns:
        Response: The HTTP response after processing.
    """
    start = time()
    response = await call_next(request)
    process_time = time() - start

    response.headers["X-Process-Time"] = str(process_time)

    if request.url.path != '/health':
        logger.info(f"Path: {request.url.path} | Method: {request.method} | Status: {response.status_code} | Time: {process_time:.4f}s")

    return response

### Security Header ###

api_key_header = APIKeyHeader(name="X-backend-secret", auto_error=False)

async def verify_secret(api_key: str = Security(api_key_header)) -> str:
    """
    Verify the provided API key against the server's secret.

    Args:
        api_key (str): The API key provided in the request header.
    
    Returns:
        str: The validated API key.
    """
    server_secret = os.getenv("BACKEND_SECRET")

    if server_secret and api_key != server_secret:
        raise HTTPException(
            status_code=403, 
            detail="Denied Access: Invalid or missing secret."
        )
    return api_key


app.include_router(query_router, prefix="/query", tags=["query"], dependencies=[Depends(verify_secret)])
app.include_router(health_router, tags=["health"])

handler = Mangum(app, lifespan='off')

if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "src.backend_api.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False  # True auto-reload for local dev. Change to False in PRODUCTION.
    )
