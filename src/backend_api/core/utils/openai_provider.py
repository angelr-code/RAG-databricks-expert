from collections.abc import AsyncGenerator

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam
from opik.integrations.openai import track_openai

from models.provider_models import ModelConfig
from src.utils.logger import get_logger

logger = get_logger()

async def generate_openai(prompt: str, config: ModelConfig, api_key: str | None) -> str: 
    """
    Generate a response from OpenAI for a given prompt and model configuration.

    Args:
        prompt (str): The input prompt.
        config (ModelConfig): The model configuration.
        api_key (str | None): The user's API key for OpenAI.
        
    Returns:
        str: The generated response text.
    """
    openai_client = AsyncOpenAI(api_key=api_key)
    tracked_openai_client = track_openai(openai_client)

    response = await tracked_openai_client.chat.completions.create(
        model=config.requested_model,
        messages=[
            ChatCompletionSystemMessageParam(role="user",content=prompt)
        ],
        temperature=config.temperature,
        max_completion_tokens=config.max_completion_tokens
    )

    return response.choices[0].message.content 

def stream_openai(prompt: str, config: ModelConfig, api_key: str | None = None) -> AsyncGenerator[str, None]:
    """
    Stream a response from OpenAI for a given prompt and model configuration.

    Args:
        prompt (str): The input prompt.
        config (ModelConfig): The model configuration.
        api_key (str | None): The user's API key for OpenAI.

    Returns:
        AsyncGenerator[str, None]: An asynchronous generator yielding response chunks.
    """
    openai_client = AsyncOpenAI(api_key=api_key)
    tracked_openai_client = track_openai(openai_client)

    async def gen() -> AsyncGenerator[str, None]:
        stream = await tracked_openai_client.chat.completions.create(
            model=config.requested_model,
            messages=[
                ChatCompletionSystemMessageParam(role="user",content=prompt)
            ],
            temperature=config.temperature,
            max_completion_tokens=config.max_completion_tokens,
            stream=True,
        )

        last_finish_reason = None
        async for chunk in stream:
            delta_text = getattr(chunk.choices[0].delta, "content", None)
            if delta_text:
                yield delta_text

            # Reasons: tool_calls, stop, length, content_filter, error
            finish_reason = getattr(chunk.choices[0], "finish_reason", None)

            if finish_reason:
                last_finish_reason = finish_reason

        logger.warning(f"Final finish_reason: {last_finish_reason}")

        # Yield a chunk to trigger truncation warning in UI
        if last_finish_reason == "length":
            yield "__truncated__"

    return gen()