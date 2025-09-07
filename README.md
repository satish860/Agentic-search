# Agentic Search

**Advanced document analysis system** with autonomous agents and TOC-guided navigation using OpenRouter API. Proven on legal contracts, ready for finance documents and beyond.

## 🎯 Core Features

- **TOC-Guided Navigation**: Smart document segmentation with GPT-5-nano for precise section targeting
- **Autonomous Agent System**: Think → Act → Observe → Complete workflow with tool calling
- **94.7% QA Accuracy**: Proven performance on legal contract analysis (up from 90.2%)
- **Document Segmentation**: Instructor-based document structure analysis with caching
- **Domain Intelligence**: Deep understanding of document patterns, cross-references, and domain-specific concepts
- **OpenRouter Integration**: GPT-5-nano for segmentation, multiple models for analysis

## 🚀 Latest Achievement: 94.7% Accuracy

### TOC-Guided Reading System
- **Document Segmentation**: Uses GPT-5-nano + Instructor for precise section identification
- **Smart Caching**: Hash-based caching prevents redundant segmentation calls
- **Intelligent Navigation**: Jumps directly to relevant sections instead of sequential reading
- **Cross-Reference Detection**: Finds related provisions across multiple document sections
- **Anti-Malformation**: Comprehensive safeguards against malformed responses

### Performance Results (Legal Contract Validation)
```
Document Type:              Legal Contracts (41 questions)
Previous Sequential:        90.2% accuracy
NEW TOC-Guided Approach:    94.7% accuracy  ⬆️ +4.5%
Malformed Responses:        0 (eliminated completely)
Perfect Matches:            94.7% of answerable questions

Next Target:                Finance Documents 🎯
```

## 🏗️ System Architecture

### Core Components
- **`src/test_sequential_reading.py`** - Main QA system with TOC-guided navigation
- **`src/document_segmenter.py`** - GPT-5-nano document segmentation with Instructor
- **`simple_agent.py`** - Base autonomous agent with tool calling system  
- **`src/llm_client.py`** - OpenRouter client with multi-model support
- **`src/config.py`** - Configuration management

### Agent Tools
- **`ReadFileTool`** - Precise line-range file reading (mimics Claude Code)
- **`DocumentSegmentTool`** - Document structure analysis with caching
- **TOC Navigation** - Smart section targeting based on question analysis

### Quick Start
```bash
# Run the enhanced QA system
cd src && uv run python test_sequential_reading.py

# Test document segmentation
cd src && uv run python document_segmenter.py

# Test agent system
uv run python simple_agent.py
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
├── src/                           # Source code
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── llm_client.py             # OpenRouter client with multi-model support
│   ├── document_segmenter.py     # GPT-5-nano document segmentation
│   ├── test_sequential_reading.py # Main QA system (94.7% accuracy)
│   └── main.py                   # Demo examples
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── test_config.py
│   └── test_llm_client.py
├── Sample/                        # Sample contract files + QA data
│   ├── LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.txt
│   └── qa_pairs.json
├── simple_agent.py               # Base autonomous agent system
├── .env                          # Environment variables (not in git)
├── .toc_cache/                   # Cached document segmentations
├── qa_results_*.json             # QA performance results
├── CLAUDE.md                     # Claude Code integration settings
├── pyproject.toml                # Project configuration
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

## 🔮 Extending to New Document Types

The TOC-guided approach is designed to work across document domains:

### Current Success: Legal Contracts ✅
- **94.7% accuracy** on contract Q&A
- **Understands**: Parties, terms, obligations, warranties, termination
- **Patterns**: Section references, legal concepts, cross-references

### Next Target: Finance Documents 🎯
- **Use Cases**: Financial reports, prospectuses, loan agreements, investment docs
- **Expected Patterns**: Financial metrics, risk factors, performance data, compliance
- **Approach**: Same TOC segmentation + domain-specific question mapping

### Future Domains 🚀
- **Technical Documentation**: API docs, manuals, specifications
- **Regulatory Documents**: Compliance filings, regulatory reports
- **Academic Papers**: Research papers, thesis documents, scientific reports

### How to Extend
1. **Add domain-specific prompts** to `TOC_GUIDED_READING_PROMPT`
2. **Update section mapping** for new document patterns  
3. **Test with domain QA data** to validate accuracy
4. **Refine cross-reference detection** for domain concepts

The core TOC-guided architecture remains the same - only the domain knowledge needs adaptation!

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