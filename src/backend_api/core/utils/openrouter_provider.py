import os

from openai import AsyncOpenAI, AuthenticationError, OpenAIError
from openai.types.chat import ChatCompletionSystemMessageParam
from opik.integrations.openai import track_openai

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
        tracked_client = track_openai(client)

        logger.info(f"Calling OpenRouter with model: {config.requested_model}")

        response = await tracked_client.chat.completions.create(
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
