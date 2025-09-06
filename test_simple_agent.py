"""Test script for Simple Agent - Step by step verification

This script tests:
1. Tool creation and basic functionality
2. XML parsing of tool calls
3. Agent loop with simple tasks
4. Integration with OpenRouter LLM
"""

from simple_agent import (
    ReadFileTool, WriteFileTool, PowerShellTool, 
    SimpleAgent, ToolCallParser, ToolCall
)
from src.llm_client import OpenRouterClient
from src.config import load_config


def test_tools_individually():
    """Test each tool individually without LLM"""
    print("Testing tools individually...")
    print("-" * 40)
    
    # Test ReadFileTool
    print("\n1. Testing ReadFileTool:")
    read_tool = ReadFileTool()
    
    # Test with existing file
    result = read_tool.execute(file_path="README.md")
    if result.success:
        print(f"SUCCESS: Read {len(result.output)} characters from README.md")
    else:
        print(f"FAILED: {result.error}")
    
    # Test with non-existent file
    result = read_tool.execute(file_path="nonexistent.txt")
    if not result.success:
        print(f"EXPECTED FAILURE: {result.error}")
    else:
        print("UNEXPECTED: Should have failed for non-existent file")
    
    # Test WriteFileTool
    print("\n2. Testing WriteFileTool:")
    write_tool = WriteFileTool()
    test_content = "This is a test file created by the simple agent test."
    
    result = write_tool.execute(file_path="test_output.txt", content=test_content)
    if result.success:
        print(f"SUCCESS: {result.output}")
    else:
        print(f"FAILED: {result.error}")
    
    # Test PowerShellTool
    print("\n3. Testing PowerShellTool:")
    ps_tool = PowerShellTool()
    
    # Test safe command
    result = ps_tool.execute(command="Get-Location")
    if result.success:
        print(f"SUCCESS: Current directory: {result.output.strip()}")
    else:
        print(f"FAILED: {result.error}")
    
    # Test dangerous command (should be blocked)
    result = ps_tool.execute(command="Remove-Item test.txt")
    if not result.success:
        print(f"EXPECTED SAFETY BLOCK: {result.error}")
    else:
        print("SAFETY CONCERN: Dangerous command was allowed!")


def test_xml_parser():
    """Test XML tool call parsing"""
    print("\n\nTesting XML Parser...")
    print("-" * 40)
    
    parser = ToolCallParser()
    
    # Test valid tool call
    test_xml = """
    I need to read a file, so I'll use the read_file tool.
    
    <tool_call>
    <name>read_file</name>
    <params>
      <file_path>contracts/sample.txt</file_path>
    </params>
    </tool_call>
    
    This should work.
    """
    
    tool_call = parser.parse_xml_tool_call(test_xml)
    if tool_call:
        print(f"SUCCESS: Parsed tool call: {tool_call.name} with params {tool_call.params}")
    else:
        print("FAILED: Could not parse valid XML tool call")
    
    # Test invalid XML
    invalid_xml = "This is just plain text without any tool calls."
    tool_call = parser.parse_xml_tool_call(invalid_xml)
    if tool_call is None:
        print("SUCCESS: Correctly returned None for text without tool calls")
    else:
        print(f"FAILED: Should not have parsed anything from plain text")


def test_simple_task_without_llm():
    """Test agent with a predefined response (no LLM call)"""
    print("\n\nTesting Agent Loop (Manual Tool Call)...")
    print("-" * 40)
    
    # Create tools and parser
    tools = [ReadFileTool(), WriteFileTool(), PowerShellTool()]
    parser = ToolCallParser()
    
    # Simulate an LLM response with a tool call
    simulated_response = """I need to list the files in the current directory.

<tool_call>
<name>powershell</name>
<params>
  <command>Get-ChildItem -Name</command>
</params>
</tool_call>"""
    
    # Parse the tool call
    tool_call = parser.parse_xml_tool_call(simulated_response)
    
    if tool_call:
        print(f"Parsed tool call: {tool_call.name}")
        
        # Find and execute the tool
        tool_dict = {tool.get_name(): tool for tool in tools}
        if tool_call.name in tool_dict:
            result = tool_dict[tool_call.name].execute(**tool_call.params)
            
            if result.success:
                print(f"Tool execution SUCCESS")
                print(f"Output: {result.output[:200]}...")
            else:
                print(f"Tool execution FAILED: {result.error}")
        else:
            print(f"Tool {tool_call.name} not found")
    else:
        print("Failed to parse tool call")


def test_with_llm():
    """Test with actual LLM integration"""
    print("\n\nTesting with LLM Integration...")
    print("-" * 40)
    
    try:
        # Load config and create LLM client
        config = load_config()
        llm_client = OpenRouterClient(config)
        
        # Create tools and agent
        tools = [ReadFileTool(), WriteFileTool(), PowerShellTool()]
        agent = SimpleAgent(llm_client, tools)
        
        # Test simple task
        print("Task: List files in current directory")
        result = agent.run("List the files in the current directory")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"LLM test failed: {e}")
        print("This might be due to missing API key or network issues")


def main():
    """Run all tests"""
    print("Simple Agent Test Suite")
    print("=" * 50)
    
    try:
        # Test 1: Individual tools
        test_tools_individually()
        
        # Test 2: XML parsing
        test_xml_parser()
        
        # Test 3: Manual agent loop
        test_simple_task_without_llm()
        
        # Test 4: LLM integration
        test_with_llm()
        
        print("\n" + "=" * 50)
        print("All tests completed!")
        
    except Exception as e:
        print(f"Test suite error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()