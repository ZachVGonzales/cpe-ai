"""
Configuration and environment setup for the pipeline.
"""

import os
import sys

# Model configuration
MODEL_ID = "gpt-5"
TEMPERATURE = 0.0
# model list: https://platform.openai.com/docs/pricing?latest-pricing=standard

# API configuration
MAX_RETRIES = 3
RETRY_DELAY = 2

# Batch API configuration
BATCH_CHECK_INTERVAL = 60  # seconds between status checks
BATCH_MAX_WAIT = 86400  # maximum wait time (24 hours)

# RAG arguments
API_VS_DOCS_DIR = "data/vector-store/raw-docs/lean_api_docs"
INFO_VS_DOCS_DIR = "data/vector-store/raw-docs/lean-info"

# Compilation configuration
LEAN_BUILD_TIMEOUT = 60  # seconds

# Default system prompt for Lean code generation
DEFAULT_SYSTEM_PROMPT = """You are an expert Lean 4 proof assistant. Convert mathematical problems into compilable Lean 4 code with parsable proof steps.

Output Format:
```lean
-- [PROBLEM_STATEMENT]
-- Original problem
-- [END_PROBLEM_STATEMENT]

set_option warningAsError true

-- Imports
import Mathlib.Data.Nat.Basic

-- [THEOREM_STATEMENT]
theorem name : statement := by
-- [END_THEOREM_STATEMENT]

  -- [PROOF]
  -- [STEP_1: Description]
  tactic
  -- [END_STEP_1]
  -- [END_PROOF]
```

Requirements:
- Must compile with `lake build`
- Must have parsable proof steps with [STEP_X:...] and [END_STEP_X] markers
- No `sorry` statements
- Use `set_option warningAsError true`
"""


def setup_environment():
    """Set up environment variables and dependencies."""
    try:
        from openai import OpenAI  # noqa: F401
    except ImportError:
        print("Error: openai package not installed. Install with: pip install openai")
        sys.exit(1)

    try:
        from dotenv import load_dotenv

        # Load environment variables from .env file
        load_dotenv()
    except ImportError:
        print(
            "Warning: python-dotenv not installed. Install with: pip install python-dotenv"
        )
        print("Continuing with system environment variables only...")


def get_api_key(api_key: str = None) -> str:
    """
    Get OpenAI API key from parameter or environment.

    Args:
        api_key: API key (if provided, takes precedence)

    Returns:
        API key string

    Raises:
        ValueError: If no API key is found
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OpenAI API key must be provided or set in OPENAI_API_KEY environment variable"
        )
    return key
