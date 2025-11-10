"""
Lean code compilation and validation utilities.
"""

import re
import subprocess
import uuid
from pathlib import Path
from typing import Tuple

from .config import LEAN_BUILD_TIMEOUT


def test_lean_compilation(lean_code: str, project_root: Path) -> Tuple[bool, str]:
    """
    Test if Lean code compiles using lake build in the project directory.

    Args:
        lean_code: The Lean code to test
        project_root: Path to the project root (where lakefile.lean is)

    Returns:
        Tuple of (success, output)
    """
    # Create temporary Lean file in the CpeAi library directory
    lib_dir = project_root / "CpeAi"
    lib_dir.mkdir(exist_ok=True)

    # Use a unique temporary filename
    temp_filename = f"TempTest_{uuid.uuid4().hex[:8]}.lean"
    temp_file = lib_dir / temp_filename

    try:
        # Write the Lean code to the temp file
        temp_file.write_text(lean_code)

        print(f"Temporary Lean file: {temp_file}")

        # Run lake build in the project root
        result = subprocess.run(
            ["lake", "build"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=LEAN_BUILD_TIMEOUT,
        )

        success = result.returncode == 0
        output = result.stdout + result.stderr

        # Log the compilation output for debugging
        print("Compilation output:")
        print(output)

        return success, output

    except subprocess.TimeoutExpired:
        print(f"Compilation timed out for file: {temp_file}")
        return False, f"Compilation timed out after {LEAN_BUILD_TIMEOUT} seconds"
    except Exception as e:
        print(f"Error running lake build for file: {temp_file}")
        print(f"Exception: {e}")
        return False, f"Error running lake build: {e}"
    finally:
        # Clean up temp file
        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception as e:
            print(f"Warning: Could not delete temp file {temp_file}: {e}")


def check_parsable_steps(lean_code: str) -> Tuple[bool, int]:
    """
    Check if Lean code has parsable proof steps.

    Args:
        lean_code: The Lean code to check

    Returns:
        Tuple of (is_parsable, step_count)
    """
    # Look for STEP markers
    step_start_pattern = r"--\s*\[STEP_\d+:"
    step_end_pattern = r"--\s*\[END_STEP_\d+\]"

    step_starts = re.findall(step_start_pattern, lean_code)
    step_ends = re.findall(step_end_pattern, lean_code)

    # Must have at least one step
    if len(step_starts) == 0 or len(step_ends) == 0:
        return False, 0

    # Number of starts and ends should match
    if len(step_starts) != len(step_ends):
        return False, len(step_starts)

    return True, len(step_starts)


def extract_lean_code(response: str) -> str:
    """
    Extract Lean code from markdown code blocks if present.

    Args:
        response: The response string from the API

    Returns:
        Extracted Lean code
    """
    # Try to find code block with lean or no language specified
    patterns = [r"```lean\n(.*?)```", r"```\n(.*?)```"]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # If no code block found, return the whole response
    return response.strip()
