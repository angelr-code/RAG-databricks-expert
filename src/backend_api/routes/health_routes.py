from time import time 
from fastapi import APIRouter

router = APIRouter()
start_time = time()

@router.get('/')
async def root():
    return {"message": "API is running"}

@router.get('/health')
async def health_check():
    up_time = int(time() - start_time)
    return {
        "up_time": up_time,
        "status": "ok"
    }
