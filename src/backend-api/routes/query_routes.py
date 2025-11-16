from fastapi import APIRouter, Request

from models.api_models import QueryRequest, QueryResponse
from core.generation_logic import generate_answer
from core.search_logic import query

router = APIRouter()

@router.post('/query', response_model=QueryResponse)
async def response_generation(request: Request, query_request: QueryRequest):
    contexts = query(request, query.query_text, query_request.keywords, query_request.limit)
    

    answer_dict = await generate_answer(
        query_text=query_request.query_text,
        contexts=contexts,
    )

    return QueryResponse(
        query_text=query_request.query_text,
        provider="openAI",
        answer=answer_dict['answer'],
        sources=answer_dict['sources'],
    )