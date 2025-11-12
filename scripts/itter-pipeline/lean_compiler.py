"""
Lean code compilation and validation utilities.
"""

import re
import subprocess
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
    # Write to the main CpeAi.lean file
    cpeai_file = project_root / "CpeAi.lean"
    
    # Backup the original file if it exists
    backup_content = None
    if cpeai_file.exists():
        backup_content = cpeai_file.read_text()

    try:
        # Write the Lean code to CpeAi.lean
        cpeai_file.write_text(lean_code)

        print(f"Testing Lean file: {cpeai_file}")

        # Run lake build to compile the entire project
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
        if success:
            print("✓ Compilation successful")
        else:
            print("✗ Compilation failed")
        
        if output.strip():
            print("Compilation output:")
            print(output)
        else:
            print("(No compilation output)")

        return success, output

    except subprocess.TimeoutExpired:
        print(f"Compilation timed out for file: {cpeai_file}")
        return False, f"Compilation timed out after {LEAN_BUILD_TIMEOUT} seconds"
    except Exception as e:
        print(f"Error running lake build for file: {cpeai_file}")
        print(f"Exception: {e}")
        return False, f"Error running lake build: {e}"
    finally:
        # Restore the original file if backup exists
        if backup_content is not None:
            try:
                cpeai_file.write_text(backup_content)
                print(f"Restored original content to {cpeai_file}")
            except Exception as e:
                print(f"Warning: Could not restore original file {cpeai_file}: {e}")


def check_parsable_steps(lean_code: str) -> Tuple[bool, int]:
    """
    Check if Lean code has parsable proof steps.
    
    This function checks for properly formatted proof steps with matching
    [STEP_X: ...] and [END_STEP_X] markers. It handles multiple theorems
    each with their own step sequences.

    Args:
        lean_code: The Lean code to check

    Returns:
        Tuple of (is_parsable, step_count)
    """
    # Look for STEP markers with step numbers
    # Pattern matches: -- [STEP_1: description] or --[STEP_1: description]
    step_start_pattern = r"--\s*\[STEP_(\d+):"
    step_end_pattern = r"--\s*\[END_STEP_(\d+)\]"

    step_starts = re.findall(step_start_pattern, lean_code)
    step_ends = re.findall(step_end_pattern, lean_code)

    # Must have at least one step
    if len(step_starts) == 0 or len(step_ends) == 0:
        return False, 0

    # Number of starts and ends should match
    if len(step_starts) != len(step_ends):
        return False, len(step_starts)
    
    # Check that each STEP_X has a matching END_STEP_X
    # We need to verify the numbers match in pairs
    # This handles multiple proofs with separate step sequences
    step_start_nums = [int(n) for n in step_starts]
    step_end_nums = [int(n) for n in step_ends]
    
    # Sort both lists to match them up
    step_start_nums.sort()
    step_end_nums.sort()
    
    # They should be identical after sorting
    if step_start_nums != step_end_nums:
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
