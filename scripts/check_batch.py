#!/usr/bin/env python3
"""
Check the status of a batch job.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.config import setup_environment, get_api_key
from pipeline.openai_client import OpenAIClient


def main():
    parser = argparse.ArgumentParser(description="Check batch status")
    parser.add_argument("batch_id", help="Batch ID to check")
    parser.add_argument("--api-key", help="OpenAI API key")

    args = parser.parse_args()

    setup_environment()
    api_key = get_api_key(args.api_key)

    client = OpenAIClient(api_key=api_key, model="gpt-4o", reasoning_effort="high")

    print(f"Checking batch: {args.batch_id}\n")

    status, info = client.check_batch_status(args.batch_id)

    print(f"\nStatus: {status}")
    print(f"\nDetailed info:")
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
