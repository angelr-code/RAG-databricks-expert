import os

from openai import AsyncOpenAI, AuthenticationError, OpenAIError
from openai.types.chat import ChatCompletionSystemMessageParam

from collections.abc import AsyncGenerator

from typing import Tuple

from src.backend_api.models.provider_models import ModelConfig
from src.utils.logger import setup_logging

logger = setup_logging()

async def generate_openrouter(prompt: str, config: ModelConfig) -> Tuple[str, str]: 
    """
    Generate a response from OpenRouter for a given prompt and model configuration.

    Args:
        prompt (str): The input prompt.
        config (ModelConfig): The model configuration.
        
    Returns:
        Tuple[str, str]: The generated response text and the model used.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OpenRouter API Key not found in environment variables.")
        raise ValueError("Configuration error: OpenRouter API Key missing.")
    try:
        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

        logger.info(f"Calling OpenRouter with model: {config.requested_model}")

        response = await client.chat.completions.create(
            model=config.requested_model,
            messages=[
                ChatCompletionSystemMessageParam(role="user",content=prompt)
            ],
            temperature=config.temperature,
            max_completion_tokens=config.max_completion_tokens
        )

        content = response.choices[0].message.content
        model_used = response.model 

        return content, model_used
    except AuthenticationError as e:
        logger.error("Authentication error with OpenRouter API.")
    except OpenAIError as e:
        logger.error(f"OpenRouter API error: {e}")

def stream_openrouter(prompt: str, config: ModelConfig) -> AsyncGenerator[str, None]:
    """
    Stream a response from OpenRouter for a given prompt and model configuration.

    Args:
        prompt (str): The input prompt.
        config (ModelConfig): The model configuration.
        api_key (str | None): The user's API key for OpenRouter.

    Returns:
        AsyncGenerator[str, None]: An asynchronous generator yielding response chunks.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1",api_key=api_key)

    async def gen() -> AsyncGenerator[str, None]:
        stream = await client.chat.completions.create(
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