from fastapi import FastAPI
from contextlib import asynccontextmanager

from utils.logger import get_logger

from db.qdrant.qdrant_client import QdrantStorage

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = get_logger()
    logger.info("Starting up the FastAPI application...")

    qdrant = QdrantStorage()
    qdrant.initialize()
    #sth before receiving requests 
    yield
    #sth right after request response 
    qdrant.close()

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

