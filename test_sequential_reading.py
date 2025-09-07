import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from simple_agent import SimpleAgent, SimpleTool, ToolResult
from src.llm_client import OpenRouterClient
from src.config import load_config


class ReadFileTool(SimpleTool):
    """
    File reading tool that mimics Claude Code's Read functionality.
    
    Features:
    - Read entire files or specific line ranges
    - 1-based line indexing (like Claude Code)
    - Line number formatting in output
    - Handles large files with offset/limit parameters
    - Graceful error handling for missing files
    """
    
    def get_name(self) -> str:
        """Get tool name"""
        return "read_file"
    
    def get_description(self) -> str:
        """Get tool description for LLM"""
        return """Tool: read_file
Description: Read a file's contents with optional line range (mimics Claude Code's Read functionality)
Usage: 
<tool_call>
<name>read_file</name>
<params>
<file_path>path/to/file.txt</file_path>
<offset>1</offset>  <!-- Optional: Starting line number (1-based) -->
<limit>50</limit>   <!-- Optional: Number of lines to read -->
</params>
</tool_call>

Features:
- Read entire files or specific line ranges  
- 1-based line indexing
- Line number formatting in output
- Handles large files with offset/limit parameters"""
    
    def execute(self, **params) -> ToolResult:
        """
        Read a file with optional line range.
        
        Args:
            **params: Dictionary containing:
                file_path: Path to the file to read
                offset: Starting line number (1-based, inclusive)
                limit: Maximum number of lines to read
            
        Returns:
            ToolResult with file contents or error
        """
        # Extract parameters
        file_path = params.get('file_path')
        offset = params.get('offset')
        limit = params.get('limit')
        
        if not file_path:
            return ToolResult(
                success=False,
                error="file_path parameter is required"
            )
        
        # Convert string parameters to integers if needed
        if offset is not None:
            try:
                offset = int(offset)
            except (ValueError, TypeError):
                offset = None
        
        if limit is not None:
            try:
                limit = int(limit)
            except (ValueError, TypeError):
                limit = None
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return ToolResult(
                    success=False,
                    error=f"File not found: {file_path}"
                )
            
            # Read the file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Apply offset and limit
            if offset is not None:
                # Convert to 0-based index
                start_idx = max(0, offset - 1)
            else:
                start_idx = 0
            
            if limit is not None:
                end_idx = start_idx + limit
            else:
                # Default to 2000 lines like Claude Code
                end_idx = start_idx + 2000
            
            # Get the selected lines
            selected_lines = lines[start_idx:end_idx]
            
            # Format with line numbers (1-based)
            formatted_lines = []
            for i, line in enumerate(selected_lines, start=start_idx + 1):
                # Remove trailing newline for formatting
                line_content = line.rstrip('\n')
                # Truncate long lines
                if len(line_content) > 2000:
                    line_content = line_content[:2000] + "..."
                # Format with line number (similar to cat -n)
                formatted_lines.append(f"{i:6d} | {line_content}")
            
            output = '\n'.join(formatted_lines)
            
            # Add summary info
            total_lines = len(lines)
            lines_read = len(selected_lines)
            
            if offset or limit:
                summary = f"\n[Read lines {start_idx + 1}-{start_idx + lines_read} of {total_lines} total]"
                output = output + summary
            
            return ToolResult(
                success=True,
                output=output
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error reading file: {str(e)}"
            )


SEQUENTIAL_READING_PROMPT = """You are a precise legal document extraction system with deep understanding of contract law concepts. Your job is to find specific legal information with legal precision and comprehensive coverage.

## Core Rules:
1. **Extract exact information with legal understanding** - recognize legal concepts even in non-standard terminology
2. **Search comprehensively** - find ALL related provisions, not just the first match
3. **Distinguish between similar legal concepts** - understand the legal differences
4. **If found**: Provide exact text with line numbers AND brief legal context
5. **If not found**: State clearly with comprehensive search confirmation

## Critical Legal Distinctions:

**Contract Dates:**
- Execution/Agreement Date = when signed ("this 7th day of...")  
- Effective Date = when operative ("effective immediately," "upon delivery," "subject to...")

**Termination Types:**
- For Cause = requires breach/specific grounds ("upon default," "failure to perform")
- For Convenience = no-fault with notice only ("may terminate," "with X days notice")
- Note: "Either party may terminate" requires analysis of WHETHER cause is required

**Rights Concepts - CRITICAL PATTERN RECOGNITION:**
- **ROFR/ROFO/ROFN** = Look for these functional equivalents:
  * "option to become" / "option to purchase" / "option to distribute"
  * "right to elect" / "right to choose" / "right to select"
  * "opportunity to" / "preference for" / "first priority"
  * "shall have the option" / "may exercise" / "election to"
  * "exclusive option" / "distributorship option"
- **License Grant** = "grants," "right to use," "permission to," "appoints...and grants"
- **Exclusivity** = "exclusive," "sole," "only from," "shall not...from any source other than"

## Enhanced Reading Strategy:
1. **Read systematically** in 100-200 line chunks depending on complexity
2. **Follow cross-references** - if you see "as provided in Section X," read that section too
3. **Look for section headers** that might contain relevant provisions
4. **Search ALL related sections** - don't stop at first match for complex topics
5. **Check defined terms** (capitalized terms) for additional context
6. **Read to the END** - legal documents often have multiple related provisions scattered throughout

## Comprehensive Search Requirements:
For each legal concept:
1. **Search entire document** - provisions may be in unexpected sections
2. **Look for functional equivalents** - legal concepts may use different terminology
3. **Consider conditional language** - "upon," "when," "if," "subject to," "provided that"
4. **Include timeframes and conditions** - "during term," "after termination," "for X years"
5. **Find exceptions and carveouts** - "except," "unless," "notwithstanding"

## Legal Context Analysis:
Before providing your answer, consider:
- **Legal PURPOSE**: Why would this provision exist in a contract?
- **Legal EFFECT**: What are the consequences?  
- **TIMING**: When does it apply (during term, after termination, immediately)?
- **CONDITIONS**: Are there "if," "upon," "subject to" qualifiers?
- **SCOPE**: Who does it apply to (parties, affiliates, successors)?

## Response Formats:

**Information Found:**
LEGAL ANALYSIS: [Brief explanation of what this provision accomplishes legally]
ANSWER: [Exact text from document]  
CITATION: "[Direct quote]" [Line: X]

**Multiple Provisions Found:**
LEGAL ANALYSIS: [Brief explanation of the overall legal framework]
ANSWER: [Comprehensive summary of all related provisions]
CITATIONS:
1. "[Provision 1]" [Line: X]
2. "[Provision 2]" [Line: Y]
3. "[Provision 3]" [Line: Z]

**Information NOT Found:**
LEGAL ANALYSIS: [Explanation of what you searched for and legal concepts considered]
ANSWER: The requested information is not found in this document.
SEARCH SUMMARY: Examined lines 1-[X], searched for [specific terms/concepts and functional equivalents], no relevant provisions located.

## Critical Instructions:
- **Think like a lawyer** - understand legal significance, not just words
- **Be thorough** - legal documents have interconnected provisions
- **Recognize patterns** - legal concepts often appear in non-obvious terminology
- **Distinguish precisely** - different legal concepts have different meanings
- **Don't give up early** - if a concept should exist, search harder for functional equivalents
- **Cross-reference sections** - follow "Section X" references to find complete information

## Special Attention Areas:
- **Price/Pricing provisions** - often scattered across multiple sections (pricing, adjustments, restrictions)
- **Post-termination obligations** - look in termination sections, survival clauses, and ongoing duty sections
- **Options/Rights** - may be described as "opportunity," "election," or "choice" rather than formal "right"
- **Assignments** - may be covered in both assignment sections AND bankruptcy/insolvency provisions

Your job: Find exactly what's asked for with legal understanding and comprehensive coverage. If legal concept exists in the document under any terminology, find it."""


def load_qa_data():
    """Load QA data and check if contract file exists"""
    print("Checking QA data and contract file...")
    
    # Check if QA file exists
    qa_file = 'Sample/qa_pairs.json'
    if os.path.exists(qa_file):
        print(f"[OK] QA file found: {qa_file}")
        try:
            with open(qa_file, 'r', encoding='utf-8', errors='ignore') as f:
                qa_data = json.load(f)
            print(f"[OK] QA data loaded: {len(qa_data)} questions")
            
            # Print first question
            if qa_data:
                first_qa = qa_data[0]
                print("\nFirst Question:")
                print(f"Q: {first_qa['question']}")
                print(f"Answers: {len(first_qa['answers'])}")
                for i, answer in enumerate(first_qa['answers'], 1):
                    print(f"  {i}. '{answer['text']}' (position: {answer['answer_start']})")
            
        except Exception as e:
            print(f"[ERROR] Failed to load QA data: {e}")
            return False
    else:
        print(f"[ERROR] QA file not found: {qa_file}")
        return False
    
    # Check if contract file exists
    contract_file = 'Sample/LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.txt'
    if os.path.exists(contract_file):
        print(f"[OK] Contract file found: {contract_file}")
        file_size = os.path.getsize(contract_file)
        print(f"[OK] Contract file size: {file_size} bytes")
    else:
        print(f"[ERROR] Contract file not found: {contract_file}")
        return False
    
    return True





def create_qa_agent_with_sequential_reading():
    """Create an agent specifically designed for QA with sequential reading"""
    try:
        config = load_config()
        llm_client = OpenRouterClient(config)
        
        # Create agent with ReadFileTool
        tools = [ReadFileTool()]
        agent = SimpleAgent(llm_client, tools)
        
        # Override the system prompt for sequential reading
        agent.custom_system_prompt = SEQUENTIAL_READING_PROMPT
        
        return agent
    except Exception as e:
        print(f"[ERROR] Failed to create QA agent: {e}")
        return None


def test_all_qa_questions():
    """Test all QA questions in the dataset"""
    print("\nTesting All QA Questions with Sequential Reading")
    print("=" * 70)
    
    # Load QA data
    try:
        with open('Sample/qa_pairs.json', 'r', encoding='utf-8', errors='ignore') as f:
            qa_data = json.load(f)
        print(f"Loaded {len(qa_data)} questions")
    except Exception as e:
        print(f"[ERROR] Failed to load QA data: {e}")
        return
    
    # Create QA agent
    agent = create_qa_agent_with_sequential_reading()
    if not agent:
        return
    
    contract_file = 'Sample/LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.txt'
    
    results = []
    
    # Process first 41 questions
    for i, qa_item in enumerate(qa_data[:41]):
        print(f"\n{'='*70}")
        print(f"QUESTION {i+1}/41")
        print(f"{'='*70}")
        
        question = qa_item['question']
        expected_answers = qa_item['answers']
        is_impossible = qa_item.get('is_impossible', False)
        
        print(f"Q: {question}")
        print(f"Expected answers: {len(expected_answers)} {'(IMPOSSIBLE)' if is_impossible else ''}")
        
        if expected_answers:
            for j, answer in enumerate(expected_answers, 1):
                print(f"  {j}. '{answer['text']}'")
        
        # Create task
        task = f"""Question: {question}

Document: {contract_file}

Find the answer by reading the document. When you find it, STOP and provide it immediately.

Start by reading the first 100 lines:
<tool_call>
<name>read_file</name>
<params>
<file_path>{contract_file}</file_path>
<offset>1</offset>
<limit>100</limit>
</params>
</tool_call>

Remember: Simple question = Simple answer."""
        
        try:
            print(f"\n[INFO] Running agent for question {i+1}...")
            result = agent.run(task)
            
            print("\nAGENT'S RESPONSE:")
            print("-" * 50)
            print(result)
            
            # Store detailed result for analysis
            results.append({
                'question_num': i+1,
                'question_id': qa_item.get('id', f"Q{i+1}"),
                'question_full': question,
                'question_short': question[:100] + "..." if len(question) > 100 else question,
                'expected_answers': [ans['text'] for ans in expected_answers],
                'expected_count': len(expected_answers),
                'is_impossible': is_impossible,
                'agent_response_full': result,
                'agent_response_short': result[:200] + "..." if len(result) > 200 else result,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"[ERROR] Failed on question {i+1}: {e}")
            results.append({
                'question_num': i+1,
                'question_id': qa_item.get('id', f"Q{i+1}"),
                'question_full': question,
                'question_short': question[:100] + "..." if len(question) > 100 else question,
                'expected_answers': [ans['text'] for ans in expected_answers],
                'expected_count': len(expected_answers),
                'is_impossible': is_impossible,
                'agent_response_full': f"ERROR: {str(e)}",
                'agent_response_short': f"ERROR: {str(e)}",
                'timestamp': datetime.now().isoformat()
            })
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"qa_results_{timestamp}.json"
    
    # Create metadata
    analysis_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_questions': len(results),
            'model_used': 'moonshotai/kimi-k2-0905',  # Current model
            'contract_file': 'Sample/LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.txt'
        },
        'results': results,
        'summary': {
            'total_questions': len(results),
            'impossible_questions': sum(1 for r in results if r['is_impossible']),
            'answerable_questions': sum(1 for r in results if not r['is_impossible']),
            'error_count': sum(1 for r in results if r['agent_response_full'].startswith('ERROR'))
        }
    }
    
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        print(f"\n[SUCCESS] Results saved to: {results_file}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save results: {e}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL 41 QUESTIONS")
    print(f"{'='*70}")
    
    for result in results:
        status = "IMPOSSIBLE" if result['is_impossible'] else f"{result['expected_count']} answers"
        print(f"Q{result['question_num']:2d}: {status:12} | {result['question_short']}")
    
    # Print analysis summary
    print(f"\n{'='*70}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"Total Questions: {len(results)}")
    print(f"Impossible Questions: {sum(1 for r in results if r['is_impossible'])}")
    print(f"Answerable Questions: {sum(1 for r in results if not r['is_impossible'])}")
    print(f"Errors: {sum(1 for r in results if r['agent_response_full'].startswith('ERROR'))}")
    print(f"Results saved to: {results_file}")
    
    return results_file



if __name__ == "__main__":
    load_qa_data()
    test_all_qa_questions()