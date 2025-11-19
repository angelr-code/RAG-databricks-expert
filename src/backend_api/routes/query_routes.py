from fastapi import APIRouter, Request, Header
from fastapi.responses import StreamingResponse

from src.backend_api.models.api_models import QueryRequest, QueryResponse
from src.backend_api.core.generation_logic import generate_answer, generate_streaming_answer

from src.backend_api.core.search_logic import search_service

router = APIRouter()

@router.post('/generate', response_model=QueryResponse)
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

    search_result = await search_service(
        query_text=query_request.query_text,
        vectorstore=vectorstore,
        limit=query_request.limit
    )

    response = await generate_answer(
        query_request=query_request,
        search_result=search_result,
        user_api_key=user_api_key
    )

    return response


@router.post('/stream')
async def stream_generation(
    request: Request, 
    query_request: QueryRequest,
    user_api_key: str | None = Header(
        None, 
        alias="OpenAI-API-Key",
        description="The user's OpenAI API key for the LLM provider"
    )
) -> StreamingResponse:
    vectorstore = request.app.state.vectorstore

    search_result = await search_service(
        query_text=query_request.query_text,
        vectorstore=vectorstore,
        limit=query_request.limit
    )

    return StreamingResponse(
        generate_streaming_answer(query_request, search_result, user_api_key),
        media_type="text/plain"
    )