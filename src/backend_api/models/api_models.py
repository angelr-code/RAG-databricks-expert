from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    query_text: str = Field(default="", description="The user input text")
    limit: int = Field(default=5, description="Number of chunks to be retrieved from the Vector DB")
    provider: str = Field(default="openai", description="External API LLM Provider")
    model: str | None = Field(default=None, description="Model to be used")

class QueryResponse(BaseModel): # need to check 
    query_text: str = Field(default="", description="The original input text")
    provider: str = Field(default="OpenRouter", description="External API LLM Provider used")
    model: str | None = Field(default=None, description="Model used")
    answer: str = Field(default="", description="Generated response")
    sources: list = Field(default_factory=[], description="List of queried sources used in the generation")
    finish_reason: str | None = Field(default=None, description="The reason why the generation finished, if available")

class SearchResult(BaseModel):
    contexts: List[str] = Field(default_factory=list, description="List of retrieved contexts from the Vector DB")
    sources: List[str] = Field(default_factory=list, description="List of source URLs associated with the contexts")