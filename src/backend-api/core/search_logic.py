import opik
from fastapi import Request

from qdrant_client.models import FieldCondition, Filter, MatchText, MatchValue

from models.api_models import QueryRequest

@opik.track
async def query(request: Request,query_text: str = "",keywords: str | None = None, limit: int = 5):
    pass