"""Tests for OpenRouter LLM client."""

import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types import CompletionUsage

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.llm_client import OpenRouterClient
from src.config import OpenRouterConfig


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return OpenRouterConfig(
        api_key="test-api-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        temperature=0.0,
        max_tokens=100
    )


@pytest.fixture
def mock_chat_completion():
    """Create a mock ChatCompletion response."""
    return ChatCompletion(
        id="test-completion-id",
        object="chat.completion",
        created=1234567890,
        model="test/model",
        choices=[
            Mock(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Test response content"
                ),
                finish_reason="stop"
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        )
    )


class TestOpenRouterClient:
    """Test cases for OpenRouterClient."""
    
    def test_init_with_config(self, mock_config):
        """Test client initialization with provided config."""
        with patch('src.llm_client.OpenAI') as mock_openai, \
             patch('src.llm_client.AsyncOpenAI') as mock_async_openai:
            
            client = OpenRouterClient(config=mock_config)
            
            assert client.config == mock_config
            mock_openai.assert_called_once_with(
                api_key="test-api-key",
                base_url="https://openrouter.ai/api/v1"
            )
            mock_async_openai.assert_called_once_with(
                api_key="test-api-key",
                base_url="https://openrouter.ai/api/v1"
            )
    
    @patch.dict(os.environ, {
        'OPENROUTER_API_KEY': 'test-env-key',
        'LITELLM_MODEL': 'test/env-model',
        'LITELLM_TEMPERATURE': '0.5'
    })
    def test_init_without_config(self):
        """Test client initialization loading config from environment."""
        with patch('src.llm_client.OpenAI') as mock_openai, \
             patch('src.llm_client.AsyncOpenAI') as mock_async_openai:
            
            client = OpenRouterClient()
            
            assert client.config.api_key == "test-env-key"
            assert client.config.model == "test/env-model"
            assert client.config.temperature == 0.5
    
    def test_chat_completion(self, mock_config, mock_chat_completion):
        """Test synchronous chat completion."""
        with patch('src.llm_client.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_chat_completion
            
            client = OpenRouterClient(config=mock_config)
            messages = [{"role": "user", "content": "Hello"}]
            
            result = client.chat_completion(messages)
            
            assert result == mock_chat_completion
            mock_client.chat.completions.create.assert_called_once_with(
                model="test/model",
                messages=messages,
                temperature=0.0,
                max_tokens=100,
                stream=False
            )
    
    def test_chat_completion_with_overrides(self, mock_config, mock_chat_completion):
        """Test chat completion with parameter overrides."""
        with patch('src.llm_client.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_chat_completion
            
            client = OpenRouterClient(config=mock_config)
            messages = [{"role": "user", "content": "Hello"}]
            
            result = client.chat_completion(
                messages,
                model="different/model",
                temperature=0.7,
                max_tokens=200
            )
            
            mock_client.chat.completions.create.assert_called_once_with(
                model="different/model",
                messages=messages,
                temperature=0.7,
                max_tokens=200,
                stream=False
            )
    
    @pytest.mark.asyncio
    async def test_async_chat_completion(self, mock_config, mock_chat_completion):
        """Test asynchronous chat completion."""
        with patch('src.llm_client.AsyncOpenAI') as mock_async_openai_class:
            mock_client = Mock()
            mock_async_openai_class.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_chat_completion)
            
            client = OpenRouterClient(config=mock_config)
            messages = [{"role": "user", "content": "Hello"}]
            
            result = await client.async_chat_completion(messages)
            
            assert result == mock_chat_completion
            mock_client.chat.completions.create.assert_called_once_with(
                model="test/model",
                messages=messages,
                temperature=0.0,
                max_tokens=100,
                stream=False
            )
    
    def test_stream_chat_completion(self, mock_config):
        """Test streaming chat completion."""
        with patch('src.llm_client.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_stream = [Mock(), Mock(), Mock()]
            mock_client.chat.completions.create.return_value = mock_stream
            
            client = OpenRouterClient(config=mock_config)
            messages = [{"role": "user", "content": "Hello"}]
            
            result = list(client.stream_chat_completion(messages))
            
            assert result == mock_stream
            mock_client.chat.completions.create.assert_called_once_with(
                model="test/model",
                messages=messages,
                temperature=0.0,
                max_tokens=100,
                stream=True
            )
    
    def test_list_models(self, mock_config):
        """Test listing available models."""
        with patch('src.llm_client.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_response = Mock()
            mock_response.data = [Mock(id="model1"), Mock(id="model2")]
            mock_client.models.list.return_value = mock_response
            
            client = OpenRouterClient(config=mock_config)
            
            result = client.list_models()
            
            assert len(result) == 2
            assert result[0].id == "model1"
            assert result[1].id == "model2"
    
    def test_analyze_contract(self, mock_config, mock_chat_completion):
        """Test contract analysis functionality."""
        with patch('src.llm_client.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_chat_completion
            
            client = OpenRouterClient(config=mock_config)
            contract_text = "This is a sample contract..."
            
            result = client.analyze_contract(contract_text)
            
            assert result == "Test response content"
            
            # Verify the call was made with appropriate prompt
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]['messages']
            assert len(messages) == 1
            assert messages[0]['role'] == 'user'
            assert contract_text in messages[0]['content']
    
    def test_analyze_contract_with_query(self, mock_config, mock_chat_completion):
        """Test contract analysis with specific query."""
        with patch('src.llm_client.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_chat_completion
            
            client = OpenRouterClient(config=mock_config)
            contract_text = "This is a sample contract..."
            query = "What are the key obligations?"
            
            result = client.analyze_contract(contract_text, query)
            
            assert result == "Test response content"
            
            # Verify the call includes the query
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]['messages']
            assert query in messages[0]['content']
    
    def test_error_handling(self, mock_config):
        """Test error handling in API calls."""
        with patch('src.llm_client.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            
            client = OpenRouterClient(config=mock_config)
            messages = [{"role": "user", "content": "Hello"}]
            
            with pytest.raises(Exception) as exc_info:
                client.chat_completion(messages)
            
            assert "OpenRouter API error" in str(exc_info.value)
            assert "API Error" in str(exc_info.value)


class TestIntegration:
    """Integration tests (require real API key)."""
    
    @pytest.mark.integration
    @patch.dict(os.environ, {
        'OPENROUTER_API_KEY': 'real-api-key-needed'
    }, clear=False)
    def test_real_api_call(self):
        """Test with real API (skipped unless OPENROUTER_API_KEY is set)."""
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key or api_key == 'real-api-key-needed':
            pytest.skip("Real API key required for integration test")
        
        client = OpenRouterClient()
        messages = [{"role": "user", "content": "Hello, this is a test."}]
        
        try:
            response = client.chat_completion(messages)
            assert response.choices[0].message.content
            assert len(response.choices[0].message.content) > 0
        except Exception as e:
            pytest.fail(f"Real API call failed: {e}")