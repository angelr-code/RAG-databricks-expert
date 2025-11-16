from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    requested_model: str = Field(default="", description="The initial model requested by user")
    #candidate_models
    #provider_sort
    #stream
    max_completion_tokens: int = Field(default=5000, description="Maximum number of tokens for completition")
    temperature: float = Field(default=0.0, description="LLM Model sampling temperature")