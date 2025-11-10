#!/usr/bin/env python3
"""
lean_compile_test.py - Test Lean file compilation

This script:
1. Takes a Lean file as input
2. Attempts to compile it using `lake build`
3. Reports any errors encountered during compilation
"""

import subprocess
import sys
from pathlib import Path


def compile_lean_file(lean_file: Path):
    """Compile a Lean file and capture errors."""
    print(f"Testing compilation for: {lean_file}")

    # Ensure the file exists
    if not lean_file.exists():
        print(f"❌ Lean file not found: {lean_file}")
        return False

    # Run lake build
    try:
        print("Running `lake build`...")
        result = subprocess.run(
            ["lake", "build"], capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            print("✅ Compilation successful")
            return True
        else:
            print("❌ Compilation failed")
            print("--- Error Output ---")
            print(result.stderr)
            print("--- End of Error Output ---")
            return False
    except Exception as e:
        print(f"❌ Error during compilation: {e}")
        return False


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 lean_compile_test.py <path-to-lean-file>")
        sys.exit(1)

    lean_file = Path(sys.argv[1])
    compile_lean_file(lean_file)


if __name__ == "__main__":
    main()
