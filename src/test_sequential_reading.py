import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from simple_agent import SimpleAgent, SimpleTool, ToolResult
from .llm_client import OpenRouterClient
from .config import load_config
from .document_segmenter import segment_document


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


class DocumentSegmentTool(SimpleTool):
    """
    Tool to segment a document into structured sections using Instructor + GPT-5-nano
    """
    
    def get_name(self) -> str:
        """Get tool name"""
        return "segment_document"
    
    def get_description(self) -> str:
        """Get tool description for LLM"""
        return """Tool: segment_document
Description: Analyze document structure and create a table of contents with sections
Usage: 
<tool_call>
<name>segment_document</name>
<params>
<file_path>path/to/document.txt</file_path>
</params>
</tool_call>

Features:
- Uses GPT-5-nano via Instructor for precise segmentation
- Caches results to avoid re-processing same documents
- Returns structured sections with line numbers
- Helps navigate large documents efficiently"""
    
    def execute(self, **params) -> ToolResult:
        """
        Segment a document into sections.
        
        Args:
            **params: Dictionary containing:
                file_path: Path to the file to segment
        
        Returns:
            ToolResult with structured section information
        """
        file_path = params.get('file_path')
        
        if not file_path:
            return ToolResult(
                success=False,
                error="file_path parameter is required"
            )
        
        try:
            # Use the segmentation function
            structured_doc = segment_document(file_path)
            
            # Format the output for the agent
            output_lines = ["Document Structure Analysis:"]
            output_lines.append("=" * 50)
            
            for i, section in enumerate(structured_doc.sections, 1):
                output_lines.append(f"{i}. {section.title}")
                output_lines.append(f"   Lines: {section.start_index} - {section.end_index}")
            
            output_lines.append("=" * 50)
            output_lines.append(f"Total sections identified: {len(structured_doc.sections)}")
            
            return ToolResult(
                success=True,
                output='\n'.join(output_lines),
                metadata={'sections': [s.model_dump() for s in structured_doc.sections]}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error segmenting document: {str(e)}"
            )


TOC_GUIDED_READING_PROMPT = """You are a precise legal document extraction system with deep understanding of contract law concepts. Your job is to find specific legal information with legal precision and comprehensive coverage using structured document navigation.

## CRITICAL OUTPUT RULES:
- NEVER return raw <tool_call> XML as your final answer
- If you're about to output a tool call, STOP and provide the actual answer instead
- Your response must ALWAYS be human-readable text with LEGAL ANALYSIS, ANSWER, and CITATIONS
- All responses must be properly formatted, never malformed

## MANDATORY WORKFLOW - FOLLOW EXACTLY:

### STEP 1: GET DOCUMENT STRUCTURE FIRST
**ALWAYS start by segmenting the document to understand its structure:**
<tool_call>
<name>segment_document</name>
<params>
<file_path>[document_path]</file_path>
</params>
</tool_call>

### STEP 2: IDENTIFY RELEVANT SECTIONS
After getting the document structure, analyze which sections are most likely to contain the answer based on:
- Question keywords and legal concepts
- Section titles and organizational structure
- Legal document patterns

### STEP 3: MULTI-PASS TARGETED READING
**CRITICAL: Use multiple search passes to ensure comprehensive coverage**

**Pass 1 - Primary Search:**
Read sections most likely to contain the answer:
<tool_call>
<name>read_file</name>
<params>
<file_path>[document_path]</file_path>
<offset>[section_start_line]</offset>
<limit>[section_length]</limit>
</params>
</tool_call>

**Pass 2 - Keyword Expansion:**
After initial findings, generate related legal terms and search again:
- Identify legal synonyms (warranty → warrants, guarantee, defect)
- Find alternative phrasings (24 month → twenty-four months)
- Look for cross-references (pursuant to Section X, as defined in)
- Search for related concepts in different sections

**Pass 3 - Cross-Reference Verification:**
Check for additional provisions in related sections:
- If found assignment clauses, also check termination sections
- If found license grants, check both appointment AND rights sections  
- If found warranty terms, check both warranty AND liability sections

## Core Rules:
1. **ALWAYS segment first** - Never start reading without understanding document structure
2. **Navigate intelligently** - Use section information to jump directly to relevant content
3. **Extract exact information with legal understanding** - recognize legal concepts even in non-standard terminology
4. **Search comprehensively** - find ALL related provisions, not just the first match
5. **If found**: Provide exact text with line numbers AND brief legal context
6. **If not found**: State clearly with comprehensive search confirmation

## MULTIPLE PROVISIONS HANDLING:
When a legal concept appears in multiple places:
1. **Don't stop at first match** - Legal documents scatter related provisions across sections
2. **Check for variations and cross-references**:
   - Assignment clauses may also appear in bankruptcy/insolvency sections
   - License grants may be split between appointment and rights sections
   - Contract provisions may include duration, procedures, obligations, and exceptions
3. **Extract ALL relevant text** - If question asks about a topic, find EVERY provision related to it
4. **Common multi-location patterns**:
   - Assignment: Look in both "Assignment" AND "Termination" sections
   - Licenses: Check "Grant", "Appointment", AND "License" sections
   - Contract terms: Include main provisions, procedures, remedies, AND limitations

## COMPREHENSIVE EXTRACTION RULES:
For complex legal questions (assignment, license grants, contract terms, etc.):
- Extract ALL related provisions, not just the main clause
- Include: duration, procedures, exceptions, obligations, remedies
- Read entire relevant sections, not just first paragraph
- Look for functional equivalents using different terminology

## CRITICAL CROSS-REFERENCES:
Always check these related sections together:
- Assignment → Also check Termination sections (bankruptcy assignments)
- License Grant → Check BOTH appointment clauses AND license sections
- Contract terms → Check main sections AND related procedural sections
- Termination → Check termination sections AND related consequence provisions

## Critical Legal Distinctions:

**Contract Dates:**
- Execution/Agreement Date = when signed ("this day of...")  
- Effective Date = when operative ("effective immediately," "upon delivery," "subject to...")

**Termination Types:**
- For Cause = requires breach/specific grounds ("upon default," "failure to perform")
- For Convenience = no-fault with notice only ("may terminate," "with notice")
- Note: "Either party may terminate" requires analysis of WHETHER cause is required

**Rights Concepts - CRITICAL PATTERN RECOGNITION:**
- **ROFR/ROFO/ROFN** = Look for these functional equivalents:
  * "option to become" / "option to purchase" / "option to distribute"
  * "right to elect" / "right to choose" / "right to select"
  * "opportunity to" / "preference for" / "first priority"
  * "shall have the option" / "may exercise" / "election to"
  * "exclusive option" / "distribution option"
- **License Grant** = "grants," "right to use," "permission to," "appoints...and grants"
- **Exclusivity** = "exclusive," "sole," "only from," "shall not...from any source other than"

## Section-to-Question Mapping Guide:
- **Document Name/Title** → Look in opening lines (1-10), title headers, and main agreement statement
- **Parties/Company info** → Look in opening paragraphs, RECITALS, signature sections
- **Dates (execution, effective)** → Opening paragraphs, term sections
- **Termination** → "TERMINATION", "DURATION", numbered termination sections
- **Payment/Pricing** → "PURCHASE", "PAYMENT", "PRICES" sections
- **Contract terms** → Main contract sections AND related procedural sections
- **Obligations** → Main numbered sections describing duties
- **Rights/Options** → "PRODUCTS", "RIGHTS", main business sections
- **Assignment** → "ASSIGNMENT", "INTERPRETATION" sections

## Enhanced Navigation Strategy:
1. **Segment document** - Get complete structural overview
2. **Map question to likely sections** - Use legal document patterns
3. **Read targeted sections** - Jump directly to relevant content
4. **Follow cross-references** - if you see "as provided in Section X," read that section too
5. **Search related sections** - legal concepts may span multiple sections
6. **Check defined terms** (capitalized terms) for additional context

## BEFORE FINALIZING YOUR ANSWER - MANDATORY CHECKLIST:
1. **Multi-Pass Verification**: Did I perform at least 2-3 search passes with different keywords?
2. **Complete Coverage**: Have I found ALL occurrences of this concept in the document?
3. **Cross-References**: Did I check related sections and follow "pursuant to Section X" references?
4. **Keyword Variations**: Did I search for legal synonyms and alternative phrasings?
5. **Multi-Part Extraction**: For complex concepts, did I extract ALL components?
6. **Proper Formatting**: Is my response properly formatted (not raw XML)?
7. **Complete Citations**: Have I provided complete citations for EVERY relevant provision?

## MULTI-PASS SEARCH EXAMPLES:
**For "Warranty Duration":**
- Pass 1: Search "warranty" sections
- Pass 2: Search "24 month", "twenty-four month", "warrants", "guarantee", "defect"
- Pass 3: Check product liability, representation sections for additional warranty terms

**For "Assignment":**
- Pass 1: Search "assignment" sections  
- Pass 2: Search "transfer", "convey", "delegate", "assign"
- Pass 3: Check termination sections for bankruptcy assignment clauses

**For "Minimum Commitment":**
- Pass 1: Search "minimum", "purchase" sections
- Pass 2: Search "units", "quarterly", "annual", "maintain exclusive", "$250,000"
- Pass 3: Check establishment sections for performance requirements

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
SEARCH SUMMARY: Segmented document into [X] sections, examined sections [list], searched for [specific terms/concepts and functional equivalents], no relevant provisions located.

## Critical Instructions:
- **MANDATORY**: Start every task with document segmentation
- **Think like a lawyer** - understand legal significance, not just words
- **Navigate strategically** - use document structure to find information efficiently
- **Be comprehensive** - don't stop at first match, find ALL related provisions
- **Extract completely** - include all aspects of multi-part legal concepts
- **Be decisive** - when you find all answers, provide them immediately and STOP
- **Don't over-search** - simple questions have simple answers in obvious places
- **Be thorough** - legal documents have interconnected provisions
- **Recognize patterns** - legal concepts often appear in non-obvious terminology
- **Cross-reference sections** - follow references to find complete information
- **Never return malformed responses** - always provide human-readable answers

## DETAILED WORKFLOW EXAMPLES:

### Example 1: Single Answer Question
**Question**: "What is the agreement date?"
**Correct Approach**:
1. Segment document first
2. Identify: Agreement dates typically in opening lines
3. Read lines 1-20 to find date
4. Provide answer with citation

**GOOD Response**:
LEGAL ANALYSIS: The agreement date establishes when the contract was executed.
ANSWER: The agreement date is September 7, 1999.
CITATION: "this 7th day of September, 1999" [Line: 8]

**BAD Response**: `<tool_call><name>read_file</name>...` (Never do this!)

### Example 2: Multiple Provisions Question  
**Question**: "What are the assignment provisions?"
**Correct Approach**:
1. Segment document first
2. Check BOTH "Assignment" section AND "Termination" section (bankruptcy clauses)
3. Extract ALL related text, not just first match
4. Provide comprehensive answer

**GOOD Response**:
LEGAL ANALYSIS: The contract contains comprehensive assignment restrictions with both consent requirements and termination triggers.
ANSWER: The contract has two assignment-related provisions: (1) consent requirement for voluntary assignments, and (2) automatic termination rights for involuntary assignments.
CITATIONS:
1. "No assignment...without prior written consent" [Line: 232]
2. "In the event either party...makes an assignment for the benefit of creditors, the other party shall be entitled to terminate" [Line: 158]

### Example 3: Complex Multi-Part Question
**Question**: "What are all warranty duration provisions?"
**Correct Approach**:
1. Segment document first
2. Find warranty section(s) 
3. Extract ALL warranty-related text: duration, procedures, obligations, exceptions
4. Don't stop at just finding "24 months" - get complete picture

**GOOD Response**:
LEGAL ANALYSIS: The warranty framework includes duration, procedures, and various obligations.
ANSWER: The warranty provisions include: (1) 24-month warranty period, (2) notice procedures, (3) repair obligations, (4) emergency repair rights, and (5) time limitations.
CITATIONS:
1. "24 months after delivery to end-user" [Line: 134]
2. "Distributor shall promptly report...within 365 days" [Line: 140]
3. "Company shall thereupon correct each such defect" [Line: 136]
[Additional citations for all aspects]

### Example 4: "Not Found" Question
**Question**: "Are there any liquidated damages clauses?"
**Correct Approach**:
1. Segment document first
2. Check termination, breach, and damages sections
3. Search comprehensively
4. Confirm not found with thorough search summary

**GOOD Response**:
LEGAL ANALYSIS: Liquidated damages clauses specify predetermined damages amounts for breaches.
ANSWER: The requested information is not found in this document.
SEARCH SUMMARY: Segmented document into 8 sections, examined termination (Section 4), breach remedies, and damages provisions, searched for "liquidated damages," "predetermined damages," "penalty," and related terms, no relevant provisions located.

## CRITICAL SUCCESS PATTERNS:
✅ Always segment first
✅ Check multiple related sections  
✅ Extract ALL relevant text
✅ Provide proper citations
✅ Never return raw XML

## CRITICAL FAILURE PATTERNS TO AVOID:
❌ Stopping at first match when multiple exist
❌ Returning raw `<tool_call>` XML
❌ Missing cross-references between sections
❌ Incomplete extraction of multi-part concepts
❌ Not checking related sections

Your job: Use structured navigation to find exactly what's asked for with legal understanding and comprehensive coverage. Always segment first, then navigate intelligently to relevant sections. Extract ALL related provisions completely. Never return malformed responses. Follow the examples above for proper execution."""


def load_qa_data():
    """Load QA data and check if contract file exists"""
    print("Checking QA data and contract file...")
    
    # Check if QA file exists
    qa_file = 'data/Sample/qa_pairs.json'
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
    contract_file = 'data/Sample/LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.txt'
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
        
        # Create agent with both ReadFileTool and DocumentSegmentTool
        tools = [ReadFileTool(), DocumentSegmentTool()]
        agent = SimpleAgent(llm_client, tools)
        
        # Override the system prompt for TOC-guided reading
        agent.custom_system_prompt = TOC_GUIDED_READING_PROMPT
        
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
        with open('data/Sample/qa_pairs.json', 'r', encoding='utf-8', errors='ignore') as f:
            qa_data = json.load(f)
        print(f"Loaded {len(qa_data)} questions")
    except Exception as e:
        print(f"[ERROR] Failed to load QA data: {e}")
        return
    
    # Create QA agent
    agent = create_qa_agent_with_sequential_reading()
    if not agent:
        return
    
    contract_file = 'data/Sample/LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.txt'
    
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

Follow the mandatory workflow: FIRST segment the document to understand its structure, THEN navigate to relevant sections to find the answer."""
        
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
            'contract_file': 'data/Sample/LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.txt'
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