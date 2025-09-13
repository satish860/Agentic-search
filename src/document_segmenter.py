"""Document Segmentation using Instructor with GPT-5-nano via OpenRouter"""

import os
import json
import hashlib
from typing import List
from pathlib import Path
import instructor
from openai import OpenAI
from pydantic import BaseModel

from config import load_config


class Section(BaseModel):
    """Section with title, start/end line numbers"""
    title: str
    start_index: int
    end_index: int


class StructuredDocument(BaseModel):
    """Document with segmented sections"""
    sections: List[Section]


def doc_with_lines(document: str) -> tuple[str, dict]:
    """Add line numbers to document"""
    document_lines = document.split("\n")
    document_with_line_numbers = ""
    line2text = {}
    for i, line in enumerate(document_lines):
        document_with_line_numbers += f"[{i}] {line}\n"
        line2text[i] = line
    return document_with_line_numbers, line2text


def segment_document(file_path: str, use_cache: bool = True, document_type: str = "legal") -> StructuredDocument:
    """Segment document into structured sections using Instructor + GPT-5-nano"""

    # Simple caching based on file hash and document type
    cache_file = f".{Path(file_path).name}.{document_type}.segments.json"

    if use_cache and os.path.exists(cache_file):
        try:
            # Check if file changed
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            current_hash = hashlib.md5(content.encode()).hexdigest()

            with open(cache_file, 'r') as f:
                cache_data = json.load(f)

            if cache_data.get('file_hash') == current_hash:
                print(f"[INFO] Using cached {document_type} segmentation for {file_path}")
                sections = [Section.model_validate(s) for s in cache_data['sections']]
                return StructuredDocument(sections=sections)

        except Exception:
            pass  # Cache invalid, regenerate

    # Read and prepare document
    print(f"[INFO] Segmenting {document_type} document with GPT-5-nano: {file_path}")

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        document = f.read()

    document_with_lines, line2text = doc_with_lines(document)

    # Set up Instructor with OpenRouter
    config = load_config()

    # Create OpenAI client pointed to OpenRouter
    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url
    )

    # Patch with instructor
    instructor_client = instructor.from_openai(client)

    # Choose prompts based on document type
    if document_type == "financial":
        system_prompt = "You are a financial document expert specializing in SEC filings (10-K, 10-Q forms). Identify main sections in financial documents."
        user_prompt = f"""Identify the main sections in this SEC financial filing (10-K form). Focus on major sections such as:

- Business Overview/Description
- Risk Factors
- Legal Proceedings
- Management's Discussion and Analysis (MD&A)
- Financial Statements (Consolidated Income Statement, Balance Sheet, Cash Flows, etc.)
- Notes to Financial Statements
- Controls and Procedures
- Exhibits

Look for standard SEC filing structure and numbered items (Item 1, Item 2, etc.).

Document:
{document_with_lines}"""
    else:  # Default to legal
        system_prompt = "You are a legal document expert. Identify main sections in contracts."
        user_prompt = f"""Identify the main sections in this legal contract. Focus only on major divisions like RECITALS and numbered sections (1., 2., 3., etc.).

Document:
{document_with_lines}"""

    try:
        # Use instructor to get structured output
        structured_doc = instructor_client.chat.completions.create(
            model="openai/gpt-5-nano",  # Using gpt-5-nano
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            response_model=StructuredDocument,
            max_tokens=8000
        )
        
        # Cache the result
        if use_cache:
            cache_data = {
                'file_hash': hashlib.md5(document.encode()).hexdigest(),
                'sections': [s.model_dump() for s in structured_doc.sections]
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        
        print(f"[SUCCESS] Document segmented into {len(structured_doc.sections)} sections using Instructor")
        return structured_doc
        
    except Exception as e:
        print(f"[ERROR] Instructor segmentation failed: {e}")
        # Return basic fallback segmentation
        return StructuredDocument(sections=[
            Section(title="Full Document", start_index=0, end_index=len(line2text)-1)
        ])


if __name__ == "__main__":
    # Test segmentation
    contract_file = 'data/Sample/LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.txt'
    
    if os.path.exists(contract_file):
        structured_doc = segment_document(contract_file)
        
        print("\nDocument Segmentation Results:")
        print("=" * 50)
        for i, section in enumerate(structured_doc.sections, 1):
            print(f"{i}. {section.title}")
            print(f"   Lines: {section.start_index} - {section.end_index}")
        print("=" * 50)
    else:
        print(f"Contract file not found: {contract_file}")