#!/usr/bin/env python3
"""
Retrieve and process results from a completed batch.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.config import setup_environment, get_api_key
from pipeline.openai_client import OpenAIClient
from pipeline.lean_compiler import (
    extract_lean_code,
    test_lean_compilation,
    check_parsable_steps,
)


def main():
    parser = argparse.ArgumentParser(description="Retrieve batch results")
    parser.add_argument("batch_id", help="Batch ID to retrieve")
    parser.add_argument(
        "--output-dir", default="batch_results", help="Directory to save results"
    )
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument(
        "--process", action="store_true", help="Process and validate Lean code"
    )

    args = parser.parse_args()

    setup_environment()
    api_key = get_api_key(args.api_key)

    client = OpenAIClient(api_key=api_key, model="gpt-4o", reasoning_effort="high")

    print(f"Retrieving batch: {args.batch_id}\n")

    # Check status first
    status, info = client.check_batch_status(args.batch_id)

    if status != "completed":
        print(f"Batch not completed. Status: {status}")
        return

    print(f"Batch completed!")
    print(f"   Total: {info['request_counts']['total']}")
    print(f"   Completed: {info['request_counts']['completed']}")
    print(f"   Failed: {info['request_counts']['failed']}\n")

    # Create output directory
    output_dir = Path(args.output_dir) / args.batch_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Retrieve results
    output_file = output_dir / f"batch_output_{args.batch_id}.jsonl"
    results = client.retrieve_batch_results(args.batch_id, output_file)

    if not results:
        print("Failed to retrieve results")
        return

    print(f"\nRetrieved {len(results)} results\n")

    # Save individual results
    for custom_id, response in results.items():
        if response:
            # Save raw response
            response_file = output_dir / f"{custom_id}_response.txt"
            response_file.write_text(response)

            # Extract Lean code
            lean_code = extract_lean_code(response)
            lean_file = output_dir / f"{custom_id}.lean"
            lean_file.write_text(lean_code)

            print(f"{custom_id}")
            print(f"   Response: {response_file}")
            print(f"   Lean code: {lean_file}")

            if args.process:
                # Test compilation
                project_root = Path(__file__).parent.parent
                compile_success, compile_output = test_lean_compilation(
                    lean_code, project_root
                )

                # Check parsability
                parsable, step_count = check_parsable_steps(lean_code)

                # Save validation results
                validation = {
                    "custom_id": custom_id,
                    "compilation": {
                        "success": compile_success,
                        "output": compile_output,
                    },
                    "parsability": {
                        "is_parsable": parsable,
                        "step_count": step_count,
                    },
                }

                validation_file = output_dir / f"{custom_id}_validation.json"
                with open(validation_file, "w") as f:
                    json.dump(validation, f, indent=2)

                status_icon = "✅" if compile_success and parsable else "❌"
                print(
                    f"   {status_icon} Compilation: {compile_success}, Parsable: {parsable} ({step_count} steps)"
                )
                print(f"   Validation: {validation_file}")

            print()
        else:
            print(f"{custom_id} - No response")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
