from collections.abc import AsyncGenerator

from typing import Tuple

from openai import AsyncOpenAI, AuthenticationError, OpenAIError
from openai.types.chat import ChatCompletionSystemMessageParam

from src.backend_api.models.provider_models import ModelConfig
from src.utils.logger import setup_logging

logger = setup_logging()

async def generate_openai(prompt: str, config: ModelConfig, api_key: str | None) -> Tuple[str, str]: 
    """
    Generate a response from OpenAI for a given prompt and model configuration.

    Args:
        prompt (str): The input prompt.
        config (ModelConfig): The model configuration.
        api_key (str | None): The user's API key for OpenAI.
        
    Returns:
        Tuple[str, str]: The generated response text and the model used.
    """
    try:
        openai_client = AsyncOpenAI(api_key=api_key)

        response = await openai_client.chat.completions.create(
            model=config.requested_model,
            messages=[
                ChatCompletionSystemMessageParam(role="user",content=prompt)
            ],
            max_completion_tokens=config.max_completion_tokens
        )

        content = response.choices[0].message.content
        model_used = response.model 

        return content, model_used
    except AuthenticationError as e:
        logger.error("Authentication error with OpenAI API. Please check your API key.")
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")

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

    async def gen() -> AsyncGenerator[str, None]:
        stream = await openai_client.chat.completions.create(
            model=config.requested_model,
            messages=[
                ChatCompletionSystemMessageParam(role="user",content=prompt)
            ],
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