"""Configuration management for Agentic Search."""

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class OpenRouterConfig(BaseModel):
    """OpenRouter API configuration."""
    
    api_key: str = Field(..., description="OpenRouter API key")
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL"
    )
    model: str = Field(
        default="anthropic/claude-sonnet-4",
        description="Default model to use"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for text generation"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens to generate"
    )


def load_config() -> OpenRouterConfig:
    """Load configuration from environment variables."""
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is required. "
            "Please set it in your .env file."
        )
    
    return OpenRouterConfig(
        api_key=api_key,
        model=os.getenv("LITELLM_MODEL", "anthropic/claude-sonnet-4"),
        temperature=float(os.getenv("LITELLM_TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
    )