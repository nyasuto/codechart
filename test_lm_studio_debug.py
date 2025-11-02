#!/usr/bin/env python3
"""Debug test script for LM Studio integration."""

import traceback

from openai import OpenAI

from src.code_chunker import CodeChunk
from src.config import Config
from src.llm_analyzer import LLMAnalyzer


def test_connection():
    """Test basic connection to LM Studio."""
    print("\n" + "=" * 60)
    print("Step 1: Testing Connection to LM Studio")
    print("=" * 60)

    config = Config.from_yaml()
    print(f"Base URL: {config.llm.base_url}")
    print(f"Model: {config.llm.model}")

    try:
        client = OpenAI(api_key=config.llm.api_key, base_url=config.llm.base_url)

        print("\nSending test request...")
        response = client.chat.completions.create(
            model=config.llm.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in one word."},
            ],
            temperature=0.3,
            max_tokens=10,
        )

        print("✅ Connection successful!")
        print(f"\nResponse object type: {type(response)}")
        print(f"Response: {response}")

        if response and response.choices:
            print(f"\nFirst choice: {response.choices[0]}")
            print(f"Message: {response.choices[0].message}")
            print(f"Content: {response.choices[0].message.content}")
        else:
            print("⚠️  Response has no choices")

        return True

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        traceback.print_exc()
        return False


def test_analyzer():
    """Test LLM analyzer."""
    print("\n" + "=" * 60)
    print("Step 2: Testing LLM Analyzer")
    print("=" * 60)

    config = Config.from_yaml()
    analyzer = LLMAnalyzer(config)

    test_code = """int add(int a, int b) {
    return a + b;
}"""

    chunk = CodeChunk(
        id="test_001",
        type="function",
        name="add",
        code=test_code,
        tokens=20,
    )

    print(f"\nCode to analyze:\n{test_code}")

    try:
        print("\nAnalyzing...")
        result = analyzer.analyze_chunk(chunk, context="Simple addition function")

        print("✅ Analysis successful!")
        print(f"\nSummary: {result.summary}")
        print(f"Purpose: {result.purpose}")
        print(f"Complexity: {result.complexity}")
        print(f"Tokens used: {result.tokens_used}")

        return True

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("LM Studio Debug Test")
    print("=" * 60)

    # Test 1: Basic connection
    if not test_connection():
        print("\n⚠️  Skipping analyzer test due to connection failure")
        return 1

    # Test 2: Analyzer
    if not test_analyzer():
        return 1

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
