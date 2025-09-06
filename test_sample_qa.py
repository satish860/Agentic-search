"""Test Simple Agent with Sample QA Comparison

This script:
1. Tests the agent with a specific question from the QA dataset
2. Compares our agent's output with the expected answer
3. Analyzes how well the agent performs on structured contract questions
"""

import json
from simple_agent import SimpleAgent, ReadFileTool, WriteFileTool, PowerShellTool
from src.llm_client import OpenRouterClient
from src.config import load_config


def load_sample_qa():
    """Load the first few QA pairs for testing"""
    with open('Sample/qa_pairs.json', 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    return qa_data[:5]  # First 5 questions for testing


def test_parties_question():
    """Test the agent with the 'Parties' question"""
    print("Testing Agent with 'Parties' Question")
    print("=" * 50)
    
    # Load configuration and create agent
    config = load_config()
    llm_client = OpenRouterClient(config)
    
    tools = [ReadFileTool(), WriteFileTool(), PowerShellTool()]
    agent = SimpleAgent(llm_client, tools)
    
    # Test question about parties
    question = """Read the contract file 'Sample/LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.txt' and identify the parties involved in this distributor agreement. 

Please provide:
1. The names of all parties mentioned
2. Their roles (e.g., Company, Distributor)  
3. Any relevant business details about each party"""
    
    print(f"Question: {question}")
    print("\n" + "-" * 50)
    
    # Run agent
    result = agent.run(question)
    
    print("\nAGENT'S ANSWER:")
    print("-" * 20)
    print(result)
    
    return result


def show_expected_answer():
    """Show the expected answer from QA dataset"""
    print("\n" + "=" * 50)
    print("EXPECTED ANSWER FROM QA DATASET:")
    print("-" * 20)
    
    qa_data = load_sample_qa()
    parties_qa = qa_data[1]  # Parties question is second in the list
    
    print(f"Question: {parties_qa['question']}")
    print(f"\nExpected Parties:")
    for answer in parties_qa['answers']:
        print(f"- {answer['text']} (position {answer['answer_start']})")


def test_agreement_date_question():
    """Test with Agreement Date question"""
    print("\n\n" + "=" * 50)
    print("Testing Agent with 'Agreement Date' Question")
    print("=" * 50)
    
    # Load configuration and create agent
    config = load_config()
    llm_client = OpenRouterClient(config)
    
    tools = [ReadFileTool(), WriteFileTool(), PowerShellTool()]
    agent = SimpleAgent(llm_client, tools)
    
    question = """Read the contract file 'Sample/LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.txt' and find the agreement date. When was this distributor agreement signed or dated?"""
    
    print(f"Question: {question}")
    print("\n" + "-" * 50)
    
    result = agent.run(question)
    
    print("\nAGENT'S ANSWER:")
    print("-" * 20)
    print(result)
    
    # Show expected answer
    qa_data = load_sample_qa()
    date_qa = qa_data[2]  # Agreement Date question
    print(f"\nEXPECTED ANSWER: {date_qa['answers'][0]['text']}")
    
    return result


def main():
    """Run QA comparison tests"""
    print("Simple Agent QA Comparison Test")
    print("=" * 60)
    
    try:
        # Test 1: Parties question
        agent_result1 = test_parties_question()
        show_expected_answer()
        
        # Test 2: Agreement Date question  
        agent_result2 = test_agreement_date_question()
        
        print("\n" + "=" * 60)
        print("COMPARISON COMPLETE!")
        print("You can now compare how well our agent performed vs expected answers")
        
    except Exception as e:
        print(f"Error running QA tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()