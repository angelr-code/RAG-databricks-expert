from fastapi import APIRouter, Request, Header

from src.backend_api.models.api_models import QueryRequest, QueryResponse
from src.backend_api.core.generation_logic import generate_answer

from src.backend_api.core.search_logic import search_service

router = APIRouter()

@router.post('/query', response_model=QueryResponse)
async def response_generation(
    request: Request, 
    query_request: QueryRequest,
    user_api_key: str | None = Header(
        None, 
        alias="OpenAI-API-Key",
        description="The user's OpenAI API key for the LLM provider"
    )
) -> QueryResponse:
    vectorstore = request.app.state.vectorstore
    embedding_model = request.app.state.embedding_model

    search_result = await search_service(
        query_text=query_request.query_text,
        vectorstore=vectorstore,
        embedding_model=embedding_model,
        keywords=query_request.keywords,
        limit=query_request.limit
    )

    response = await generate_answer(
        query_req=query_request,
        search_result=search_result,
        user_api_key=user_api_key
    )

    return response