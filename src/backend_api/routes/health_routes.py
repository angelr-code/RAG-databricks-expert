from time import time 
from fastapi import APIRouter, Request

from qdrant_client.http.exceptions import UnexpectedResponse

router = APIRouter()
start_time = time()

@router.get('/')
async def root():
    """Root endpoint.

    Returns a JSON indicating that the API is running.
    """
    return {"message": "API is running"}

@router.get('/health')
async def health_check():
    """
    Liveness check nedpoint
    """
    up_time = int(time() - start_time)
    return {
        "up_time": up_time,
        "status": "ok"
    }

@router.get('/ready')
async def api_readiness(request: Request):
    """
    Readiness checkpoint 

    Verifies whether Qdrant is ready to handle requests
    """
    try:
        vectorestore = request.app.vectorstore

        await vectorestore.client.get_collections() # Light check
        return {"status": "ready"}
    except UnexpectedResponse: 
        return {"status": "not ready", "reason": "Qdrant unexpected response"}
    except Exception as e:
        return {"status": "not ready", "reason": f"{e}"}
