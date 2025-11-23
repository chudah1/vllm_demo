"""
Example client for the vLLM API server.
Demonstrates how to use the OpenAI-compatible API for completions and chat.
"""

from openai import OpenAI
import sys


def setup_client(base_url: str = "http://localhost:8000/v1", api_key: str = "dummy"):
    """
    Set up the OpenAI client pointing to our vLLM server.

    Args:
        base_url: Base URL of the vLLM API server
        api_key: API key (use 'dummy' if no authentication is set)

    Returns:
        OpenAI client instance
    """
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )


def example_completion(client: OpenAI):
    """
    Example of basic text completion.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Text Completion")
    print("=" * 60)

    response = client.completions.create(
        model="default",  # Model name is ignored, uses configured model
        prompt="Once upon a time in a land far away,",
        max_tokens=100,
        temperature=0.7,
    )

    print(f"\nPrompt: Once upon a time in a land far away,")
    print(f"\nCompletion:\n{response.choices[0].text}")
    print(f"\nTokens used: {response.usage.total_tokens}")


def example_chat(client: OpenAI):
    """
    Example of chat completion.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Chat Completion")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the three laws of robotics?"},
    ]

    response = client.chat.completions.create(
        model="default",
        messages=messages,
        max_tokens=200,
        temperature=0.7,
    )

    print(f"\nUser: {messages[1]['content']}")
    print(f"\nAssistant: {response.choices[0].message.content}")
    print(f"\nTokens used: {response.usage.total_tokens}")


def example_streaming_completion(client: OpenAI):
    """
    Example of streaming text completion.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Streaming Text Completion")
    print("=" * 60)

    print("\nPrompt: Write a haiku about programming")
    print("\nStreaming response:")

    stream = client.completions.create(
        model="default",
        prompt="Write a haiku about programming",
        max_tokens=50,
        temperature=0.8,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].text:
            print(chunk.choices[0].text, end="", flush=True)

    print("\n")


def example_streaming_chat(client: OpenAI):
    """
    Example of streaming chat completion.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Streaming Chat Completion")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function to calculate factorial."},
    ]

    print(f"\nUser: {messages[1]['content']}")
    print("\nAssistant (streaming): ", end="", flush=True)

    stream = client.chat.completions.create(
        model="default",
        messages=messages,
        max_tokens=200,
        temperature=0.7,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n")


def example_with_parameters(client: OpenAI):
    """
    Example showing different parameter configurations.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Different Temperature Settings")
    print("=" * 60)

    prompt = "The future of AI is"

    print(f"\nPrompt: {prompt}")

    # Low temperature (more deterministic)
    print("\n--- Low Temperature (0.2) - More Deterministic ---")
    response_low = client.completions.create(
        model="default",
        prompt=prompt,
        max_tokens=50,
        temperature=0.2,
    )
    print(response_low.choices[0].text)

    # High temperature (more creative)
    print("\n--- High Temperature (1.5) - More Creative ---")
    response_high = client.completions.create(
        model="default",
        prompt=prompt,
        max_tokens=50,
        temperature=1.5,
    )
    print(response_high.choices[0].text)


def example_conversation(client: OpenAI):
    """
    Example of multi-turn conversation.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Multi-turn Conversation")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
    ]

    # First turn
    print(f"\nUser: {messages[-1]['content']}")
    response1 = client.chat.completions.create(
        model="default",
        messages=messages,
        max_tokens=100,
    )
    assistant_message1 = response1.choices[0].message.content
    print(f"Assistant: {assistant_message1}")

    # Add assistant response to conversation
    messages.append({"role": "assistant", "content": assistant_message1})

    # Second turn
    messages.append({"role": "user", "content": "Can you give me a simple code example?"})
    print(f"\nUser: {messages[-1]['content']}")

    response2 = client.chat.completions.create(
        model="default",
        messages=messages,
        max_tokens=150,
    )
    assistant_message2 = response2.choices[0].message.content
    print(f"Assistant: {assistant_message2}")


def main():
    """
    Main function to run all examples.
    """
    # Parse command line arguments
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/v1"

    print(f"Connecting to vLLM API server at: {base_url}")

    # Set up client
    client = setup_client(base_url)

    try:
        # Check server health
        print("\nChecking server health...")
        # Note: Health endpoint is not part of OpenAI API, so we use requests
        import requests
        health_url = base_url.replace("/v1", "/health")
        health_response = requests.get(health_url)
        print(f"Server status: {health_response.json()}")

        # Run examples
        example_completion(client)
        example_chat(client)
        example_streaming_completion(client)
        example_streaming_chat(client)
        example_with_parameters(client)
        example_conversation(client)

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure the vLLM API server is running at the specified URL.")
        sys.exit(1)


if __name__ == "__main__":
    main()
