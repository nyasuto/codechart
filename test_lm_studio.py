#!/usr/bin/env python3
"""Simple test script for LM Studio integration."""

from src.code_chunker import CodeChunk
from src.config import Config
from src.llm_analyzer import LLMAnalyzer


def main():
    """Test LLM analyzer with a simple code chunk."""
    print("=" * 60)
    print("LM Studio Integration Test")
    print("=" * 60)

    # Load configuration
    print("\n1. Loading configuration...")
    config = Config.from_yaml()
    print(f"   Provider: {config.llm.provider}")
    print(f"   Base URL: {config.llm.base_url}")
    print(f"   Model: {config.llm.model}")

    # Create analyzer
    print("\n2. Creating LLM analyzer...")
    analyzer = LLMAnalyzer(config)

    # Create a simple test chunk
    print("\n3. Creating test code chunk...")
    test_code = """int add(int a, int b) {
    return a + b;
}"""

    chunk = CodeChunk(
        id="test_001",
        type="function",
        name="add",
        code=test_code,
        tokens=20,
        metadata={"file_name": "test.c"},
    )
    print(f"   Code:\n{test_code}")

    # Analyze the chunk
    print("\n4. Analyzing code with LM Studio...")
    print("   (This may take a few seconds...)")

    try:
        result = analyzer.analyze_chunk(chunk, context="Simple addition function")

        print("\n" + "=" * 60)
        print("Analysis Result")
        print("=" * 60)
        print(f"\nChunk ID: {result.chunk_id}")
        print(f"Chunk Name: {result.chunk_name}")
        print(f"\n【Summary】\n{result.summary}")
        print(f"\n【Purpose】\n{result.purpose}")
        print(f"\n【Algorithm】\n{result.algorithm}")
        print(f"\n【Complexity】\n{result.complexity}")
        print(f"\n【Dependencies】")
        for dep in result.dependencies:
            print(f"  - {dep}")
        print(f"\n【Potential Issues】")
        for issue in result.potential_issues:
            print(f"  - {issue}")
        print(f"\n【Improvements】")
        for imp in result.improvements:
            print(f"  - {imp}")
        print(f"\n【Tokens Used】\n{result.tokens_used}")

        print("\n" + "=" * 60)
        print("✅ Test completed successfully!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ Test failed!")
        print("=" * 60)
        print(f"\nError: {e}")
        print("\nPlease check:")
        print("  1. LM Studio is running")
        print("  2. Model 'openai/gpt-oss-20b' is loaded")
        print("  3. Server is listening on http://127.0.0.1:1234")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
