#!/usr/bin/env python3
"""
process_opc.py - Process OPC JSON through OpenAI API and test Lean code (Iterative Version)

This script:
1. Reads a JSON file with math problems
2. Uses iterative pipeline with tool calls for step-by-step proof generation
3. Tests if generated Lean code compiles at each step
4. Checks if code has parsable proof steps
5. Saves successful outputs to a dataset directory

Uses the iterative pipeline which allows the model to:
- Search Lean documentation via file_search
- Read workspace state via read_file_state
- Apply code patches via apply_patch
- Iteratively fix compilation errors up to NO_COMPILE_LIMIT
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import pipeline module
sys.path.insert(0, str(Path(__file__).parent))

from itter_pipeline.config import MODEL_ID, setup_environment, get_api_key
from itter_pipeline.processor import LeanCodeProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Process OPC JSON through OpenAI API and test Lean code"
    )
    parser.add_argument("input_file", help="Path to input JSON file with problems")
    parser.add_argument(
        "--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--model", default=MODEL_ID, help=f"OpenAI model to use (default: {MODEL_ID})"
    )
    parser.add_argument("--system-prompt", help="Path to system prompt file (optional)")
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="Directory to save successful outputs (default: dataset)",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        help="Maximum number of problems to process (default: all)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="Maximum tokens for OpenAI API response (default: 4000)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        choices=["low", "medium", "high"],
        help="Reasoning effort level for reasoning models (default: high)",
    )
    parser.add_argument(
        "--no-skip-geometry",
        action="store_true",
        help="Don't skip complex geometry problems (default: skip them)",
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        default=False,
        help="Use RAG service with vector store retrieval (Note: iterative pipeline uses built-in file_search)",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG (for compatibility, iterative pipeline always has file_search available)",
    )

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    try:
        # Set up environment (loads .env, checks dependencies)
        setup_environment()

        # Get API key
        api_key = get_api_key(args.api_key)

        # Note: Iterative pipeline always uses file_search tool for documentation
        # The use_rag flag is kept for compatibility but file_search is built-in
        use_rag = not args.no_rag if hasattr(args, 'no_rag') else args.use_rag
        
        print("\n" + "="*80)
        print("ITERATIVE PIPELINE MODE")
        print("="*80)
        print("This pipeline uses OpenAI tool calling for iterative proof generation:")
        print("  - file_search: Searches Lean documentation")
        print("  - read_file_state: Reads current workspace files")
        print("  - apply_patch: Applies code changes and tests compilation")
        print("  - Iteratively fixes errors up to NO_COMPILE_LIMIT")
        print("="*80 + "\n")

        processor = LeanCodeProcessor(
            api_key=api_key,
            model=args.model,
            system_prompt_file=args.system_prompt,
            dataset_dir=args.dataset_dir,
            reasoning_effort=args.reasoning_effort,
            skip_complex_geometry=not args.no_skip_geometry,
            use_rag=use_rag,
        )

        # Process using iterative pipeline (no batch mode in iterative version)
        processor.process_json_file(args.input_file, max_problems=args.max_problems)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
