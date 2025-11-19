import opik

import json

from collections.abc import AsyncGenerator

from src.backend_api.models.api_models import SearchResult, QueryRequest, QueryResponse
from src.backend_api.models.provider_models import ModelConfig
from src.utils.logger import setup_logging
from src.backend_api.core.utils.openai_provider import generate_openai, stream_openai
from src.backend_api.core.utils.openrouter_provider import generate_openrouter, stream_openrouter

logger = setup_logging()

PROMPT = """
You are a skilled technical assistant specialized in Databricks.
Respond to the user’s query using the provided context from these articles and pieces of documentation,
which are retrieved from a vector database, without relying on outside knowledge or assumptions.


### Output Rules:
- Write a detailed, structured answer using **Markdown** (headings, bullet points,
  short or long paragraphs as appropriate).
- Use up to **{tokens} tokens** without exceeding this limit.
- Only include facts from the provided context from the articles.
- Attribute each fact to the correct author(s) and source, and include **clickable links**.
- If the article author and feed author differ, mention both.
- There is no need to mention that you based your answer on the provided context.
- But if no relevant information exists, clearly state this and provide a fallback suggestion.

### Query:
{query}

### Context Articles:
{context_texts}

### Final Answer:
"""

def build_prompt(query_text: str, contexts: SearchResult, max_tokens: int) -> str:
    """
    Build the prompt for the LLM using the query text and retrieved contexts.
    Args:
        query_text (str): The user's query text.
        contexts (SearchResult): The retrieved contexts from the vector database.
        max_tokens (int): The maximum number of tokens for the response.
        
    Returns:
        str: The formatted prompt string.
    """
    contexts_list = [
        f"- URL: {source}\n Content: {context}" for source, context in zip(contexts.sources, contexts.contexts)
    ]
    contexts = "\n\n".join(contexts_list)

    return PROMPT.format(
        query = query_text,
        context_texts=contexts,
        tokens=max_tokens
    )

@opik.track(name="generate_answer")
async def generate_answer(query_request: QueryRequest, search_result: SearchResult, user_api_key: str | None = None) -> QueryResponse:
    """
    Generate an answer based on the query request and search results.
    
    Args:
        query_request (QueryRequest): The user's query request.
        search_result (SearchResult): The search results from the vector database.
        user_api_key (str | None): The user's API key for the LLM provider.
    
    Returns:
        QueryResponse: The generated response including the answer and metadata.
    """

    selected_model = query_request.model or "gpt-4o-mini"
    config = ModelConfig(requested_model=selected_model)

    prompt = build_prompt(query_request.query_text, search_result, config.max_completion_tokens)
    
    if query_request.provider == "openai":
        logger.info("OpenAI provider selected for generation...")
        answer_text, model_used = await generate_openai(prompt, config, user_api_key)
    elif query_request.provider == "OpenRouter":
        logger.info("OpenRouter provider selected for generation...")
        answer_text, model_used = await generate_openrouter(prompt, config)     
    else:
        logger.warning(f"Unsupported provider: {query_request.provider}. Selecting ")

    response = QueryResponse(
        query_text=query_request.query_text,
        provider=query_request.provider,
        model=model_used,
        answer=answer_text,
        sources=list(search_result.sources)
    )

    return response

async def generate_streaming_answer(query_request: QueryRequest, search_result: SearchResult, user_api_key: str | None = None) -> AsyncGenerator[str,None]:
    """
    Generate a streaming answer based on the query request and search results.
    
    Args:
        query_request (QueryRequest): The user's query request.
        search_result (SearchResult): The search results from the vector database.
        user_api_key (str | None): The user's API key for the LLM provider.
    
    Yields:
        AsyncGenerator[str, None]: An asynchronous generator yielding response chunks.
    """
    selected_model = query_request.model or "gpt-4o-mini"
    config = ModelConfig(requested_model=selected_model)

    prompt = build_prompt(query_request.query_text, search_result, config.max_completion_tokens)

    sources_data = json.dumps({
        "type": "sources", 
        "data": list(search_result.sources)
    })
    yield f"{sources_data}\n"

    generator = None
    if query_request.provider == "openai":
        logger.info("OpenAI provider selected for generation...")
        generator = stream_openai(prompt, config, user_api_key)
    elif query_request.provider == "OpenRouter":
        logger.info("OpenRouter provider selected for generation...")
        generator = stream_openrouter(prompt, config)   
    else:
        logger.warning(f"Unsupported provider: {query_request.provider}. Selecting ")
        return 
    
    async for chunk in generator:
        yield chunk