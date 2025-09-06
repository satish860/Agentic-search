"""Simple Agentic Loop with Tool Calling - Single File Prototype

This file demonstrates:
1. Basic tool definition and execution
2. LLM integration for decision making  
3. Simple agentic loop (think → act → observe → repeat)
4. XML-based tool call parsing
5. Context management across iterations
"""

import os
import re
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import xml.etree.ElementTree as ET

# Import existing LLM client
from src.llm_client import OpenRouterClient
from src.config import load_config


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    output: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ToolCall:
    """Parsed tool call from LLM response"""
    name: str
    params: Dict[str, Any]


class SimpleTool(ABC):
    """Base class for all tools"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get tool name"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get tool description for LLM"""
        pass
    
    @abstractmethod
    def execute(self, **params) -> ToolResult:
        """Execute the tool with given parameters"""
        pass


class ReadFileTool(SimpleTool):
    """Tool to read file contents"""
    
    def get_name(self) -> str:
        return "read_file"
    
    def get_description(self) -> str:
        return """Read contents of a file.
        Parameters:
        - file_path: Path to the file to read
        
        Usage:
        <tool_call>
        <name>read_file</name>
        <params>
          <file_path>path/to/file.txt</file_path>
        </params>
        </tool_call>"""
    
    def execute(self, **params) -> ToolResult:
        file_path = params.get('file_path')
        if not file_path:
            return ToolResult(success=False, output="", error="file_path parameter required")
        
        try:
            path = Path(file_path)
            if not path.exists():
                return ToolResult(success=False, output="", error=f"File not found: {file_path}")
            
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            return ToolResult(
                success=True,
                output=content,
                metadata={"file_size": len(content), "file_path": str(path)}
            )
        
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class WriteFileTool(SimpleTool):
    """Tool to write content to a file"""
    
    def get_name(self) -> str:
        return "write_file"
    
    def get_description(self) -> str:
        return """Write content to a file.
        Parameters:
        - file_path: Path to the file to write
        - content: Content to write to the file
        
        Usage:
        <tool_call>
        <name>write_file</name>
        <params>
          <file_path>path/to/file.txt</file_path>
          <content>Content to write</content>
        </params>
        </tool_call>"""
    
    def execute(self, **params) -> ToolResult:
        file_path = params.get('file_path')
        content = params.get('content', '')
        
        if not file_path:
            return ToolResult(success=False, output="", error="file_path parameter required")
        
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return ToolResult(
                success=True,
                output=f"Successfully wrote {len(content)} characters to {file_path}",
                metadata={"file_path": str(path), "content_length": len(content)}
            )
        
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class PowerShellTool(SimpleTool):
    """Tool to execute PowerShell commands (read-only operations)"""
    
    def get_name(self) -> str:
        return "powershell"
    
    def get_description(self) -> str:
        return """Execute a PowerShell command (read-only operations only).
        Parameters:
        - command: The PowerShell command to execute
        
        Usage:
        <tool_call>
        <name>powershell</name>
        <params>
          <command>Get-ChildItem -Path .</command>
        </params>
        </tool_call>"""
    
    def execute(self, **params) -> ToolResult:
        command = params.get('command')
        if not command:
            return ToolResult(success=False, output="", error="command parameter required")
        
        # Safety check - only allow read-only commands
        dangerous_commands = ['Remove-Item', 'rm', 'del', 'rmdir', 'format', 'Move-Item', 'Copy-Item']
        if any(cmd in command for cmd in dangerous_commands):
            return ToolResult(success=False, output="", error="Dangerous command not allowed")
        
        try:
            result = subprocess.run(
                ['powershell', '-Command', command],
                capture_output=True,
                text=True,
                timeout=30,
                shell=False
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR: {result.stderr}"
            
            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error=None if result.returncode == 0 else f"Command failed with return code {result.returncode}",
                metadata={"return_code": result.returncode, "command": command}
            )
        
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Command timed out after 30 seconds")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class ToolCallParser:
    """Parse tool calls from LLM responses"""
    
    @staticmethod
    def parse_xml_tool_call(text: str) -> Optional[ToolCall]:
        """Parse XML-formatted tool call from LLM response"""
        try:
            # Extract tool_call XML block
            pattern = r'<tool_call>(.*?)</tool_call>'
            match = re.search(pattern, text, re.DOTALL)
            if not match:
                return None
            
            xml_content = f"<tool_call>{match.group(1)}</tool_call>"
            root = ET.fromstring(xml_content)
            
            # Extract tool name
            name_elem = root.find('name')
            if name_elem is None:
                return None
            tool_name = name_elem.text
            
            # Extract parameters
            params = {}
            params_elem = root.find('params')
            if params_elem is not None:
                for param in params_elem:
                    params[param.tag] = param.text or ""
            
            return ToolCall(name=tool_name, params=params)
        
        except ET.ParseError:
            return None
        except Exception:
            return None


class SimpleAgent:
    """Simple agent with agentic loop and tool calling"""
    
    def __init__(self, llm_client: OpenRouterClient, tools: List[SimpleTool]):
        self.llm = llm_client
        self.tools = {tool.get_name(): tool for tool in tools}
        self.context = []
        self.max_iterations = 10
        self.parser = ToolCallParser()
    
    def get_system_prompt(self) -> str:
        """Get system prompt with tool descriptions"""
        tool_descriptions = "\n\n".join([tool.get_description() for tool in self.tools.values()])
        
        return f"""You are an AI assistant that can use tools to complete tasks. You have access to the following tools:

{tool_descriptions}

When you need to use a tool, format your response with the tool call in XML format as shown in the tool descriptions.

You work in an iterative loop:
1. Think about what you need to do next
2. If you need to use a tool, make a tool call
3. Observe the result
4. Continue until the task is complete

If you don't need to use any tools, just provide your final answer directly.

Always be helpful and complete the user's request to the best of your ability."""
    
    def think(self, task: str, iteration: int) -> str:
        """Ask LLM to think about next action"""
        # Build context from previous iterations
        context_str = ""
        if self.context:
            context_str = "\n\nContext from previous actions:\n"
            for i, ctx in enumerate(self.context):
                context_str += f"Step {i+1}: {ctx}\n"
        
        prompt = f"""Task: {task}

Iteration: {iteration + 1}/{self.max_iterations}
{context_str}

What should I do next to complete this task? If you need to use a tool, make a tool call. If the task is complete, provide your final answer."""
        
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.chat_completion(messages)
        return response.choices[0].message.content
    
    def is_task_complete(self, response: str) -> bool:
        """Check if the task appears to be complete"""
        # Simple heuristic: if there's no tool call and the response looks final
        has_tool_call = '<tool_call>' in response
        completion_indicators = [
            'task is complete',
            'final answer',
            'summary:',
            'conclusion:',
            'in conclusion',
            'to summarize'
        ]
        
        has_completion_indicator = any(indicator in response.lower() for indicator in completion_indicators)
        
        return not has_tool_call or has_completion_indicator
    
    def run(self, task: str) -> str:
        """Run the agentic loop to complete the task"""
        print(f"Starting task: {task}")
        print("=" * 60)
        
        for iteration in range(self.max_iterations):
            print(f"\nIteration {iteration + 1}: Thinking...")
            
            # Think about next action
            thought = self.think(task, iteration)
            print(f"Agent thought: {thought[:200]}...")
            
            # Check if task is complete
            if self.is_task_complete(thought):
                print(f"Task completed after {iteration + 1} iterations!")
                return thought
            
            # Try to parse tool call
            tool_call = self.parser.parse_xml_tool_call(thought)
            if tool_call is None:
                print("No valid tool call found, treating as final answer")
                return thought
            
            print(f"Using tool: {tool_call.name} with params: {tool_call.params}")
            
            # Execute tool
            if tool_call.name not in self.tools:
                error_msg = f"Unknown tool: {tool_call.name}"
                print(f"ERROR: {error_msg}")
                self.context.append(f"ERROR: {error_msg}")
                continue
            
            result = self.tools[tool_call.name].execute(**tool_call.params)
            
            if result.success:
                print(f"Tool executed successfully")
                print(f"Output: {result.output[:200]}...")
                self.context.append(f"Used {tool_call.name}: SUCCESS - {result.output}")
            else:
                print(f"Tool execution failed: {result.error}")
                self.context.append(f"Used {tool_call.name}: FAILED - {result.error}")
        
        print(f"Maximum iterations ({self.max_iterations}) reached")
        return "Task did not complete within maximum iterations. Please try breaking it down into smaller steps."


def main():
    """Test the simple agent with contract reading"""
    print("Testing Simple Agent with Contract Reading")
    print("=" * 60)
    
    try:
        # Load configuration and initialize LLM client
        config = load_config()
        llm_client = OpenRouterClient(config)
        
        # Initialize tools
        tools = [
            ReadFileTool(),
            WriteFileTool(),
            PowerShellTool()
        ]
        
        # Create agent
        agent = SimpleAgent(llm_client, tools)
        
        # Test 1: List contract files
        print("\nTest 1: List contract files")
        result1 = agent.run("List all files in the contracts directory")
        print(f"Result: {result1}")
        
        # Test 2: Read and analyze a contract
        print("\nTest 2: Read and analyze a contract")
        result2 = agent.run("Read the first contract file you can find in the contracts directory and provide a brief summary of what type of contract it is and key parties involved")
        print(f"Result: {result2}")
        
        # Test 3: Create a summary file
        print("\nTest 3: Create a summary file")
        result3 = agent.run("Create a file called 'contract_summary.txt' with a summary of any contract you've analyzed")
        print(f"Result: {result3}")
        
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()