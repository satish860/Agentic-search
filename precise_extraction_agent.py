"""Precise Text Extraction Agent - Simple extension to find exact answers

This adds one focused tool to the simple agent for finding exact text spans
that match the QA dataset format for lawyer highlighting.
"""

import json
import re
from typing import List, Dict
from simple_agent import SimpleTool, ToolResult, SimpleAgent, ReadFileTool, WriteFileTool, PowerShellTool
from src.llm_client import OpenRouterClient
from src.config import load_config


class FindExactTextTool(SimpleTool):
    """Tool to find exact text spans in a document"""
    
    def get_name(self) -> str:
        return "find_exact_text"
    
    def get_description(self) -> str:
        return """Find exact text spans in a file for lawyer highlighting.
        Parameters:
        - file_path: Path to the file to search
        - search_terms: List of exact terms to find (comma-separated)
        
        Returns all occurrences with their positions.
        
        Usage:
        <tool_call>
        <name>find_exact_text</name>
        <params>
          <file_path>Sample/contract.txt</file_path>
          <search_terms>Company,Distributor,Electric City Corp.</search_terms>
        </params>
        </tool_call>"""
    
    def execute(self, **params) -> ToolResult:
        file_path = params.get('file_path')
        search_terms = params.get('search_terms', '')
        
        if not file_path:
            return ToolResult(success=False, output="", error="file_path parameter required")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse search terms
            terms = [term.strip() for term in search_terms.split(',') if term.strip()]
            
            results = []
            
            for term in terms:
                # Find all occurrences of this exact term
                # Try case-sensitive first
                positions = []
                start = 0
                while True:
                    pos = content.find(term, start)
                    if pos == -1:
                        break
                    positions.append(pos)
                    start = pos + 1
                
                # If no case-sensitive matches, try case-insensitive
                if not positions:
                    pattern = re.escape(term)
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        positions.append(match.start())
                
                if positions:
                    for pos in positions:
                        results.append({
                            'text': term,
                            'answer_start': pos,
                            'context': content[max(0, pos-50):pos+len(term)+50].replace('\n', ' ')
                        })
            
            # Sort by position
            results.sort(key=lambda x: x['answer_start'])
            
            output = {
                'file': file_path,
                'total_matches': len(results),
                'answers': results
            }
            
            return ToolResult(
                success=True,
                output=json.dumps(output, indent=2),
                metadata=output
            )
            
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


def test_precise_extraction():
    """Test precise extraction to match QA expected results"""
    print("Testing Precise Text Extraction")
    print("=" * 60)
    
    # Load expected answers from QA dataset
    with open('Sample/qa_pairs.json', 'r', encoding='utf-8', errors='ignore') as f:
        qa_data = json.load(f)
    
    # Initialize tool
    tool = FindExactTextTool()
    
    # Test 1: Extract parties
    print("\n1. PARTIES EXTRACTION")
    print("-" * 30)
    
    parties_qa = qa_data[1]
    expected_parties = [ans['text'] for ans in parties_qa['answers']]
    
    print("Expected parties to find:")
    for party in expected_parties:
        print(f"  - {party}")
    
    # Search for all expected parties
    result = tool.execute(
        file_path='Sample/LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.txt',
        search_terms=','.join(expected_parties)
    )
    
    if result.success:
        found_data = json.loads(result.output)
        print(f"\nFound {found_data['total_matches']} matches:")
        
        # Group by text
        by_text = {}
        for match in found_data['answers']:
            text = match['text']
            if text not in by_text:
                by_text[text] = []
            by_text[text].append(match['answer_start'])
        
        for text, positions in by_text.items():
            print(f"  '{text}': found at positions {positions}")
        
        # Check if we found all expected parties
        expected_set = set(expected_parties)
        found_set = set(by_text.keys())
        
        if expected_set == found_set:
            print("\nSUCCESS: Found ALL expected parties!")
        else:
            missing = expected_set - found_set
            if missing:
                print(f"\nMissing: {missing}")
            extra = found_set - expected_set
            if extra:
                print(f"Extra: {extra}")
    
    # Test 2: Extract dates
    print("\n2. DATE EXTRACTION")
    print("-" * 30)
    
    date_qa = qa_data[2]
    expected_dates = [ans['text'] for ans in date_qa['answers']]
    
    print("Expected dates to find:")
    for date in expected_dates:
        print(f"  - {date}")
    
    result = tool.execute(
        file_path='Sample/LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.txt',
        search_terms=','.join(expected_dates)
    )
    
    if result.success:
        found_data = json.loads(result.output)
        print(f"\nFound {found_data['total_matches']} matches:")
        for match in found_data['answers']:
            print(f"  '{match['text']}' at position {match['answer_start']}")
    
    # Test 3: Integration with SimpleAgent
    print("\n3. INTEGRATION TEST WITH SIMPLE AGENT")
    print("-" * 30)
    
    try:
        config = load_config()
        llm_client = OpenRouterClient(config)
        
        # Add our precise tool to the simple agent
        tools = [
            ReadFileTool(),
            FindExactTextTool(),
            WriteFileTool(),
            PowerShellTool()
        ]
        
        agent = SimpleAgent(llm_client, tools)
        
        # Ask agent to find parties using the exact text tool
        task = """Use the find_exact_text tool to find these exact parties in the file 'Sample/LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.txt':
        - Distributor
        - Electric City Corp.
        - Electric City of Illinois L.L.C.
        - Company
        - Electric City of Illinois LLC
        
        Return the positions where each party appears."""
        
        print("Running agent with task...")
        result = agent.run(task)
        print(f"Agent result: {result[:500]}...")
        
    except Exception as e:
        print(f"Agent test error: {e}")


def compare_with_qa_dataset():
    """Direct comparison with QA dataset answers"""
    print("\n" + "=" * 60)
    print("DIRECT QA COMPARISON")
    print("=" * 60)
    
    # Load QA data
    with open('Sample/qa_pairs.json', 'r', encoding='utf-8', errors='ignore') as f:
        qa_data = json.load(f)
    
    # Load contract
    with open('Sample/LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.txt', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Check first 3 questions
    for i, qa in enumerate(qa_data[:3]):
        print(f"\nQuestion {i+1}: {qa['question'][:50]}...")
        print("-" * 30)
        
        for answer in qa['answers']:
            text = answer['text']
            expected_pos = answer['answer_start']
            
            # Check if text exists at expected position
            actual_text = content[expected_pos:expected_pos+len(text)]
            
            if actual_text == text:
                print(f"FOUND: '{text}' at position {expected_pos}")
            else:
                # Find where it actually is
                actual_pos = content.find(text)
                if actual_pos != -1:
                    print(f"MOVED: '{text}' expected at {expected_pos}, found at {actual_pos}")
                else:
                    print(f"NOT FOUND: '{text}' (expected at {expected_pos})")


if __name__ == "__main__":
    test_precise_extraction()
    compare_with_qa_dataset()