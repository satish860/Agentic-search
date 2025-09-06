"""Tests for configuration module."""

import pytest
import os
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import OpenRouterConfig, load_config


class TestOpenRouterConfig:
    """Test cases for OpenRouterConfig model."""
    
    def test_config_with_required_fields(self):
        """Test config creation with required fields."""
        config = OpenRouterConfig(api_key="test-key")
        
        assert config.api_key == "test-key"
        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.model == "openrouter/anthropic/claude-sonnet-4"
        assert config.temperature == 0.0
        assert config.max_tokens is None
    
    def test_config_with_all_fields(self):
        """Test config creation with all fields."""
        config = OpenRouterConfig(
            api_key="test-key",
            base_url="https://custom.api.com/v1",
            model="custom/model",
            temperature=0.7,
            max_tokens=1000
        )
        
        assert config.api_key == "test-key"
        assert config.base_url == "https://custom.api.com/v1"
        assert config.model == "custom/model"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
    
    def test_config_validation_temperature_range(self):
        """Test temperature validation."""
        # Valid temperature
        config = OpenRouterConfig(api_key="test", temperature=1.5)
        assert config.temperature == 1.5
        
        # Invalid temperature (too high)
        with pytest.raises(ValueError):
            OpenRouterConfig(api_key="test", temperature=2.5)
        
        # Invalid temperature (too low)
        with pytest.raises(ValueError):
            OpenRouterConfig(api_key="test", temperature=-0.5)


class TestLoadConfig:
    """Test cases for load_config function."""
    
    @patch.dict(os.environ, {
        'OPENROUTER_API_KEY': 'test-api-key',
    }, clear=True)
    def test_load_config_minimal(self):
        """Test loading config with minimal environment variables."""
        config = load_config()
        
        assert config.api_key == "test-api-key"
        assert config.model == "openrouter/anthropic/claude-sonnet-4"
        assert config.temperature == 0.0
        assert config.max_tokens is None
    
    @patch.dict(os.environ, {
        'OPENROUTER_API_KEY': 'test-api-key',
        'LITELLM_MODEL': 'custom/model',
        'LITELLM_TEMPERATURE': '0.8',
        'MAX_TOKENS': '512'
    }, clear=True)
    def test_load_config_complete(self):
        """Test loading config with all environment variables."""
        config = load_config()
        
        assert config.api_key == "test-api-key"
        assert config.model == "custom/model"
        assert config.temperature == 0.8
        assert config.max_tokens == 512
    
    @patch.dict(os.environ, {}, clear=True)
    def test_load_config_missing_api_key(self):
        """Test loading config without API key raises error."""
        with pytest.raises(ValueError) as exc_info:
            load_config()
        
        assert "OPENROUTER_API_KEY environment variable is required" in str(exc_info.value)
    
    @patch.dict(os.environ, {
        'OPENROUTER_API_KEY': 'test-api-key',
        'LITELLM_TEMPERATURE': 'invalid'
    }, clear=True)
    def test_load_config_invalid_temperature(self):
        """Test loading config with invalid temperature."""
        with pytest.raises(ValueError):
            load_config()
    
    @patch.dict(os.environ, {
        'OPENROUTER_API_KEY': 'test-api-key',
        'MAX_TOKENS': 'invalid'
    }, clear=True)
    def test_load_config_invalid_max_tokens(self):
        """Test loading config with invalid max_tokens."""
        with pytest.raises(ValueError):
            load_config()