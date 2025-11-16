from collections.abc import AsyncGenerator

import opik

from models.api_models import SearchResult
from models.provider_models import ModelConfig
from src.utils.logger import get_logger
from core.utils.openai_provider import generate_openai, stream_openai

logger = get_logger()

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

def build_prompt(query_text: str, contexts: SearchResult, tokens: int = 5000):
    contexts = "\n\n".join(
        (
            f"- URL: {document.source}"
            f" Content: {document.context}"
        ) for document in contexts
    )

    return PROMPT.format(
        query = query_text,
        context_texts=contexts,
        tokens=tokens
    )

@opik.track(name="generate_answer")
async def generate_answer(query_text: str, contexts: SearchResult, provider: str = "openai", selected_model: str | None = None):
    prompt = build_prompt(query_text, contexts)
    config = ModelConfig(primary_model="gpt-4o-mini")

    answer, model_used = await generate_openai(prompt, config=config)

    return {
        "answer": answer,
        "sources": [document.url for document in contexts],
        "model": model_used,
    }
