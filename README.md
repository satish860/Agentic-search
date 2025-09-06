# Agentic Search

LLM-powered contract analysis with **autonomous agent system** using OpenRouter API.

## Features

- **Agentic Loop**: Think → Act → Observe → Complete workflow for autonomous task execution
- **Tool Calling**: XML-based tool execution integrated with LLM decision making
- **Contract Analysis**: Autonomous analysis of 398+ contract files with multi-step reasoning
- **Text Extraction**: Precise text span extraction for lawyer highlighting with character positions
- **Multi-Agent Capability**: Sequential tool execution with context management
- **OpenRouter Integration**: Uses OpenRouter's API for access to multiple LLM models
- **OpenAI SDK Compatible**: Built using the OpenAI SDK for easy integration
- **Async Support**: Full async/await support for non-blocking operations
- **Streaming**: Real-time streaming responses for better user experience
- **Comprehensive Testing**: Full test suite with mocks and integration tests

## Autonomous Agent System

### Working Components
- `simple_agent.py` - Main agent with agentic loop and tool calling system
- `precise_extraction_agent.py` - Specialized text extraction for QA matching
- `test_simple_agent.py` - Comprehensive test suite for agent functionality
- `test_sample_qa.py` - QA validation tests with sample contracts

### Proven Capabilities
- **Autonomous Contract Analysis**: Reads contracts and generates detailed summaries automatically
- **QA Text Extraction**: Successfully finds ALL expected text spans from structured QA datasets
- **Tool Integration**: ReadFile, WriteFile, PowerShell, and TextSearch tools working seamlessly
- **Position-Accurate Results**: Provides exact character positions for lawyer document highlighting
- **Multi-Step Reasoning**: Handles complex tasks requiring multiple tool calls and decision points

### Quick Test
```bash
# Test the agentic loop with contract analysis
uv run python simple_agent.py

# Test precise text extraction
uv run python precise_extraction_agent.py

# Test QA comparison
uv run python test_sample_qa.py
```

## Installation

### Prerequisites

- Python 3.13 or higher
- uv package manager (recommended)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/satish860/Agentic-search.git
   cd Agentic-search
   ```

2. **Install dependencies:**
   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -e .
   ```

3. **Install test dependencies:**
   ```bash
   # Using uv
   uv sync --extra test

   # Or using pip
   pip install -e .[test]
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenRouter API key
   ```

### Environment Configuration

Create a `.env` file with the following variables:

```env
# Required: OpenRouter API Key
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: Model configuration (defaults shown)
LITELLM_MODEL=openrouter/anthropic/claude-sonnet-4
LITELLM_TEMPERATURE=0.0

# Optional: Token limit
MAX_TOKENS=4096
```

## Usage

### Basic Example

```python
from src.llm_client import OpenRouterClient

# Initialize client (loads config from .env)
client = OpenRouterClient()

# Simple chat completion
messages = [
    {"role": "user", "content": "What is machine learning?"}
]

response = client.chat_completion(messages)
print(response.choices[0].message.content)
```

### Contract Analysis

```python
# Analyze a contract
contract_text = "Your contract text here..."
analysis = client.analyze_contract(contract_text)
print(analysis)

# Analyze with specific query
query = "What are the key obligations of each party?"
analysis = client.analyze_contract(contract_text, query)
print(analysis)
```

### Streaming Responses

```python
# Stream responses for real-time output
for chunk in client.stream_chat_completion(messages):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
```

### Async Operations

```python
import asyncio

async def main():
    # Async chat completion
    response = await client.async_chat_completion(messages)
    print(response.choices[0].message.content)
    
    # Async contract analysis
    analysis = await client.async_analyze_contract(contract_text)
    print(analysis)

asyncio.run(main())
```

### Running the Demo

```bash
python main.py
```

This will run various demos including:
- Basic chat completion
- Contract analysis (using sample contracts)
- Streaming responses
- Async operations
- Model listing

## Project Structure

```
agentic-search/
├── src/                    # Source code
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── llm_client.py      # Main OpenRouter client
│   └── main.py            # Demo examples
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── test_config.py
│   └── test_llm_client.py
├── contracts/              # Sample contract files
├── .env                   # Environment variables (not in git)
├── .gitignore
├── main.py                # Entry point
├── pyproject.toml         # Project configuration
└── README.md
```

## Testing

### Run all tests:
```bash
pytest
```

### Run specific test files:
```bash
pytest tests/test_llm_client.py
pytest tests/test_config.py
```

### Run with coverage:
```bash
pytest --cov=src
```

### Run integration tests (requires real API key):
```bash
pytest -m integration
```

## API Client Features

### OpenRouterClient Methods

- `chat_completion(messages, **kwargs)` - Synchronous chat completion
- `async_chat_completion(messages, **kwargs)` - Async chat completion
- `stream_chat_completion(messages, **kwargs)` - Streaming chat completion
- `async_stream_chat_completion(messages, **kwargs)` - Async streaming
- `list_models()` - List available models
- `async_list_models()` - Async list models
- `analyze_contract(text, query="")` - Contract analysis
- `async_analyze_contract(text, query="")` - Async contract analysis

### Configuration Options

- `api_key`: OpenRouter API key (required)
- `base_url`: API base URL (default: https://openrouter.ai/api/v1)
- `model`: Default model to use
- `temperature`: Generation temperature (0.0-2.0)
- `max_tokens`: Maximum tokens to generate

## OpenRouter Models

This client supports any model available through OpenRouter, including:
- OpenAI models (GPT-4, GPT-3.5, etc.)
- Anthropic models (Claude)
- Open source models (Llama, Mixtral, etc.)
- Specialized models for different tasks

Check available models:
```python
models = client.list_models()
for model in models:
    print(f"- {model.id}")
```

## Error Handling

The client includes comprehensive error handling:

```python
try:
    response = client.chat_completion(messages)
except Exception as e:
    print(f"API error: {e}")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for your changes
4. Run the test suite
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Create an issue on GitHub
- Check OpenRouter documentation: https://openrouter.ai/docs