"""Main example demonstrating OpenRouter LLM client usage."""

from .llm_client import OpenRouterClient
from .config import load_config


def demo_basic_chat():
    """Demonstrate basic chat completion."""
    print("=== Basic Chat Completion Demo ===")
    
    try:
        # Initialize client
        client = OpenRouterClient()
        
        # Simple chat
        messages = [
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        response = client.chat_completion(messages)
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")
        print(f"Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")
        
    except Exception as e:
        print(f"Error in basic chat demo: {e}")


def main():
    """Main function to run the demo."""
    print("OpenRouter LLM Client Demo")
    print("=" * 50)
    
    # Check configuration
    try:
        config = load_config()
        print(f"Configuration loaded successfully!")
        print(f"Model: {config.model}")
        print(f"Temperature: {config.temperature}")
        print(f"Base URL: {config.base_url}")
        print()
    except Exception as e:
        print(f"Configuration error: {e}")
        print("Please check your .env file and ensure OPENROUTER_API_KEY is set.")
        return
    
    # Run basic chat demo
    demo_basic_chat()
    
    print("\n" + "=" * 50)
    print("Demo completed!")


if __name__ == "__main__":
    main()