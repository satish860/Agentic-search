# FinanceBench Integration Guide

## Overview

FinanceBench is a comprehensive financial document Q&A evaluation dataset created by Patronus AI, containing **10,231 questions** across **368 financial documents** from publicly traded companies (2015-2023). This dataset serves as a benchmark for evaluating LLM performance on financial document analysis.

## Dataset Components

### Location
```
src/datasets/finance_bench/
├── financebench_open_source.jsonl          # 10,231 questions (909 KB)
└── financebench_document_information.jsonl # 368 document metadata (87 KB)
```

### 1. Questions Dataset (`financebench_open_source.jsonl`)

**Key Fields:**
- `financebench_id` - Unique question identifier
- `company` - Company name (e.g., "3M", "APPLE", "MICROSOFT")
- `doc_name` - Document reference (e.g., "3M_2018_10K")
- `question` - **The actual question to be answered**
- `answer` - Expected/correct answer with detailed justification
- `question_type` - Question category
- `question_reasoning` - Cognitive skill being tested
- `evidence` - Pre-extracted relevant text from documents with page numbers

**Question Types:**
- **metrics-generated** - Automated extraction of specific financial metrics
- **domain-relevant** - Expert-crafted questions requiring financial analysis
- **novel-generated** - AI-generated questions for broader coverage

**Reasoning Categories:**
- Information extraction
- Logical reasoning (based on numerical reasoning)
- Numerical reasoning
- Mixed reasoning types

### 2. Document Metadata (`financebench_document_information.jsonl`)

**Key Fields:**
- `doc_name` - Document identifier matching questions file
- `company` - Company name
- `gics_sector` - Industry sector (e.g., "Industrials", "Technology")
- `doc_type` - Document type ("10k", "10q", "8k")
- `doc_period` - Year (2015-2023)
- `doc_link` - Direct URL to original SEC filing PDF

## Integration Strategy

### Current System Advantages
- **Proven TOC-guided navigation** (94.7% accuracy on legal contracts)
- **Domain-agnostic architecture** - Can handle financial documents
- **Smart caching system** - Avoid redundant processing
- **Document segmentation** - GPT-5-nano for structure analysis

### Recommended Approach: **Lightweight Integration**

#### ✅ Do This:
1. **Use JSONL files only** - Questions already contain pre-extracted evidence text
2. **On-demand PDF fetching** - Access documents via direct GitHub URLs when needed
3. **Smart caching** - Cache processed financial segments in `.finance_cache/`
4. **Incremental testing** - Start with subset, then scale

#### ❌ Avoid This:
- Downloading all 368 PDFs (unnecessary storage overhead)
- Processing full documents when evidence is pre-extracted
- Ignoring existing TOC-guided approach

## Implementation Roadmap

### Phase 1: Dataset Integration
```python
# Load dataset
import json
import pandas as pd

# Questions
with open('src/datasets/finance_bench/financebench_open_source.jsonl', 'r') as f:
    questions = [json.loads(line) for line in f]

# Document metadata  
with open('src/datasets/finance_bench/financebench_document_information.jsonl', 'r') as f:
    doc_info = [json.loads(line) for line in f]

# Merge for full context
df_questions = pd.DataFrame(questions)
df_docs = pd.DataFrame(doc_info)
df_full = pd.merge(df_questions, df_docs, on="doc_name")
```

### Phase 2: Finance Evaluation Module
- Adapt `test_sequential_reading.py` for financial documents
- Create `src/finance_bench_eval.py` 
- Implement financial document-specific prompts
- Handle numerical precision and financial terminology

### Phase 3: Performance Benchmarking
- Compare against 94.7% legal contract baseline
- Measure accuracy across question types
- Analyze performance by company sector
- Generate comprehensive evaluation reports

## Code Examples

### Basic Dataset Loading
```python
def load_financebench_subset(limit=100):
    """Load first N questions for testing"""
    questions = []
    with open('src/datasets/finance_bench/financebench_open_source.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            questions.append(json.loads(line))
    return questions

# Test with 10 questions first
test_questions = load_financebench_subset(10)
```

### Evaluation Loop Structure
```python
def evaluate_finance_questions(questions, agent):
    """Evaluate agent on finance questions"""
    results = []
    
    for q in questions:
        # Extract question details
        question = q['question']
        expected_answer = q['answer']
        company = q['company']
        doc_name = q['doc_name']
        
        # Get agent response
        agent_answer = agent.answer_question(
            question=question,
            document_context=doc_name,
            company=company
        )
        
        # Compare and score
        score = calculate_accuracy(agent_answer, expected_answer)
        results.append({
            'question_id': q['financebench_id'],
            'expected': expected_answer,
            'actual': agent_answer,
            'score': score
        })
    
    return results
```

### On-Demand PDF Access
```python
def get_pdf_url(doc_name):
    """Get direct GitHub URL for PDF"""
    base_url = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"
    return f"{base_url}/{doc_name}.pdf"

# Example: Access 3M 2018 10-K when needed
pdf_url = get_pdf_url("3M_2018_10K")
```

## Performance Targets

Based on existing legal contract results:
- **Current Legal Baseline**: 94.7% accuracy (41 questions)
- **FinanceBench Target**: 85%+ accuracy (financial documents are more complex)
- **Question Type Goals**:
  - Information extraction: 90%+
  - Numerical reasoning: 80%+
  - Logical reasoning: 75%+

## Key Considerations

### Financial Document Characteristics
- **Numerical precision** matters (e.g., "$1,577" vs "$1577.00")
- **Multiple data sources** in single document (balance sheet, income statement, cash flow)
- **Temporal comparisons** (FY2022 vs FY2021)
- **Complex financial terminology** and abbreviations

### Technical Requirements
- Handle large financial documents (10-K reports can be 100+ pages)
- Maintain compatibility with OpenRouter API + GPT-5-nano
- Preserve caching for efficiency
- Support both exact matches and semantic similarity for answers

## Next Steps

1. **Start Small**: Test with 10-20 questions to validate approach
2. **Adapt Prompts**: Modify existing legal prompts for financial context  
3. **Build Incrementally**: Scale up gradually while maintaining accuracy
4. **Document Results**: Track performance improvements and challenges

This approach leverages your proven TOC-guided system while expanding into financial document analysis, maintaining the core innovation that achieved 94.7% accuracy on legal contracts.