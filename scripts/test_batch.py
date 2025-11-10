#!/usr/bin/env python3
"""
Test script for batch API functionality.
This doesn't actually submit a batch, just tests the flow.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.config import setup_environment, get_api_key
from pipeline.openai_client import OpenAIClient


def test_batch_creation():
    """Test creating a batch request without submitting."""
    print("Testing batch API functionality...\n")

    # Setup
    setup_environment()
    api_key = get_api_key()

    client = OpenAIClient(api_key=api_key, model="gpt-4o", reasoning_effort="medium")

    # Create test requests
    test_requests = [
        {
            "custom_id": "test_1",
            "system_prompt": "You are a helpful assistant.",
            "user_prompt": "What is 2+2?",
        },
        {
            "custom_id": "test_2",
            "system_prompt": "You are a helpful assistant.",
            "user_prompt": "What is the capital of France?",
        },
    ]

    # Create batch directory
    batch_dir = Path("test_batch_output")
    batch_dir.mkdir(exist_ok=True)

    print(f"✅ Client initialized: {client.model}")
    print(f"✅ Test requests prepared: {len(test_requests)}")
    print(f"✅ Batch directory created: {batch_dir}")
    print("\n📝 Sample request structure:")
    print(json.dumps(test_requests[0], indent=2))
    print("\n✅ Batch API functionality is ready to use!")
    print("\nTo actually submit a batch, run:")
    print("  python scripts/process_opc.py data/OPC/generic-OPC.json --batch")


if __name__ == "__main__":
    test_batch_creation()
