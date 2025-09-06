"""OpenRouter LLM Client using OpenAI SDK."""

import asyncio
from typing import AsyncGenerator, List, Optional, Dict, Any
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types import Model

from .config import OpenRouterConfig, load_config


class OpenRouterClient:
    """OpenRouter LLM client using OpenAI SDK for compatibility."""
    
    def __init__(self, config: Optional[OpenRouterConfig] = None):
        """Initialize the OpenRouter client.
        
        Args:
            config: OpenRouter configuration. If None, loads from environment.
        """
        self.config = config or load_config()
        
        # Initialize OpenAI client with OpenRouter configuration
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
        
        # Initialize async client
        self.async_client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> ChatCompletion:
        """Create a chat completion.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            model: Model to use. Defaults to config model.
            temperature: Temperature for generation. Defaults to config temperature.
            max_tokens: Maximum tokens to generate. Defaults to config max_tokens.
            stream: Whether to stream the response.
            **kwargs: Additional parameters passed to the API.
            
        Returns:
            ChatCompletion response from the API.
        """
        try:
            response = self.client.chat.completions.create(
                model=model or self.config.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                stream=stream,
                **kwargs
            )
            return response
        except Exception as e:
            raise Exception(f"OpenRouter API error: {str(e)}") from e
    
    async def async_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> ChatCompletion:
        """Async version of chat completion.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            model: Model to use. Defaults to config model.
            temperature: Temperature for generation. Defaults to config temperature.
            max_tokens: Maximum tokens to generate. Defaults to config max_tokens.
            stream: Whether to stream the response.
            **kwargs: Additional parameters passed to the API.
            
        Returns:
            ChatCompletion response from the API.
        """
        try:
            response = await self.async_client.chat.completions.create(
                model=model or self.config.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                stream=stream,
                **kwargs
            )
            return response
        except Exception as e:
            raise Exception(f"OpenRouter API error: {str(e)}") from e
    
    def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Stream a chat completion.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            model: Model to use. Defaults to config model.
            temperature: Temperature for generation. Defaults to config temperature.
            max_tokens: Maximum tokens to generate. Defaults to config max_tokens.
            **kwargs: Additional parameters passed to the API.
            
        Yields:
            ChatCompletionChunk objects as they arrive.
        """
        try:
            stream = self.client.chat.completions.create(
                model=model or self.config.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                yield chunk
                
        except Exception as e:
            raise Exception(f"OpenRouter API error: {str(e)}") from e
    
    async def async_stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Async stream a chat completion.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            model: Model to use. Defaults to config model.
            temperature: Temperature for generation. Defaults to config temperature.
            max_tokens: Maximum tokens to generate. Defaults to config max_tokens.
            **kwargs: Additional parameters passed to the API.
            
        Yields:
            ChatCompletionChunk objects as they arrive.
        """
        try:
            stream = await self.async_client.chat.completions.create(
                model=model or self.config.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                yield chunk
                
        except Exception as e:
            raise Exception(f"OpenRouter API error: {str(e)}") from e
    
    def list_models(self) -> List[Model]:
        """List available models.
        
        Returns:
            List of available models.
        """
        try:
            response = self.client.models.list()
            return response.data
        except Exception as e:
            raise Exception(f"OpenRouter API error: {str(e)}") from e
    
    async def async_list_models(self) -> List[Model]:
        """Async list available models.
        
        Returns:
            List of available models.
        """
        try:
            response = await self.async_client.models.list()
            return response.data
        except Exception as e:
            raise Exception(f"OpenRouter API error: {str(e)}") from e
    
    def analyze_contract(self, contract_text: str, query: str = "") -> str:
        """Analyze a contract using the LLM.
        
        Args:
            contract_text: The contract text to analyze.
            query: Optional specific query about the contract.
            
        Returns:
            Analysis result as a string.
        """
        if query:
            prompt = f"""Analyze the following contract and answer the specific query.

Query: {query}

Contract:
{contract_text}

Please provide a detailed analysis addressing the query."""
        else:
            prompt = f"""Analyze the following contract and provide key insights:

Contract:
{contract_text}

Please provide:
1. Contract type and parties involved
2. Key terms and conditions
3. Important dates and deadlines
4. Potential risks or concerns
5. Summary of main obligations"""
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        response = self.chat_completion(messages)
        return response.choices[0].message.content
    
    async def async_analyze_contract(self, contract_text: str, query: str = "") -> str:
        """Async analyze a contract using the LLM.
        
        Args:
            contract_text: The contract text to analyze.
            query: Optional specific query about the contract.
            
        Returns:
            Analysis result as a string.
        """
        if query:
            prompt = f"""Analyze the following contract and answer the specific query.

Query: {query}

Contract:
{contract_text}

Please provide a detailed analysis addressing the query."""
        else:
            prompt = f"""Analyze the following contract and provide key insights:

Contract:
{contract_text}

Please provide:
1. Contract type and parties involved
2. Key terms and conditions
3. Important dates and deadlines
4. Potential risks or concerns
5. Summary of main obligations"""
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        response = await self.async_chat_completion(messages)
        return response.choices[0].message.content