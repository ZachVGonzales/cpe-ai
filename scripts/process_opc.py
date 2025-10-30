#!/usr/bin/env python3
"""
process_opc.py - Process OPC JSON through OpenAI API and test Lean code

This script:
1. Reads a JSON file with math problems
2. Sends each problem to OpenAI API with the appropriate system prompt
3. Tests if generated Lean code compiles
4. Checks if code has parsable proof steps
5. Saves successful outputs to a dataset directory
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime

try:
    from openai import OpenAI
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


class LeanCodeProcessor:
    """Process math problems through OpenAI API and test Lean code"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        system_prompt_file: Optional[str] = None,
        lean_check_script: str = "scripts/data/lean-check.py",
        dataset_dir: str = "dataset",
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ):
        """
        Initialize the processor

        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: OpenAI model to use
            system_prompt_file: Path to system prompt file
            lean_check_script: Path to lean-check.py script
            dataset_dir: Directory to save successful outputs
            temperature: Temperature for OpenAI API
            max_tokens: Maximum tokens for OpenAI API response
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY environment variable"
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.lean_check_script = lean_check_script

        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(dataset_dir) / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Store run metadata
        self.run_metadata = {
            "timestamp": timestamp,
            "start_time": datetime.now().isoformat(),
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt_file": system_prompt_file,
        }

        print(f"📁 Run directory: {self.run_dir}")

        # Load system prompt
        if system_prompt_file:
            with open(system_prompt_file, "r") as f:
                self.system_prompt = f.read()
        else:
            # Default system prompt if none provided
            self.system_prompt = self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt"""
        return """You are an expert Lean 4 proof assistant. Convert mathematical problems into compilable Lean 4 code with parsable proof steps.

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

    def process_problem(self, problem: Dict) -> Tuple[bool, Optional[str], Dict]:
        """
        Process a single problem through OpenAI API and test the result

        Args:
            problem: Dictionary containing problem data

        Returns:
            Tuple of (success, lean_code, metadata)
        """
        start_time = time.time()
        problem_id = problem.get("problem_id", "unknown")
        problem_text = problem.get("problem", "")

        print(f"\n{'='*80}")
        print(f"Processing problem: {problem_id}")
        print(f"{'='*80}")

        # Initialize problem log
        problem_log = {
            "problem_id": problem_id,
            "problem": problem_text,
            "start_time": datetime.now().isoformat(),
            "solution": problem.get("solution", ""),
            "metadata": problem.get("metadata", {}),
        }

        # Create prompt from problem
        user_prompt = self._create_user_prompt(problem)
        problem_log["user_prompt"] = user_prompt

        # Call OpenAI API
        print("Calling OpenAI API...")
        api_start = time.time()
        lean_code = self._call_openai_api(user_prompt)
        api_duration = time.time() - api_start

        problem_log["api_call"] = {
            "duration_seconds": round(api_duration, 2),
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if not lean_code:
            print("❌ Failed to get response from OpenAI API")
            problem_log["status"] = "failed"
            problem_log["error"] = "No response from API"
            problem_log["end_time"] = datetime.now().isoformat()
            problem_log["total_duration_seconds"] = round(time.time() - start_time, 2)
            return False, None, problem_log

        # Extract Lean code from response
        lean_code = self._extract_lean_code(lean_code)
        problem_log["generated_code"] = lean_code

        # Test if code compiles
        print("\nTesting Lean code compilation...")
        compile_start = time.time()
        compile_success, compile_output = self._test_lean_compilation(lean_code)
        compile_duration = time.time() - compile_start

        problem_log["compilation"] = {
            "success": compile_success,
            "duration_seconds": round(compile_duration, 2),
            "output": compile_output,
        }

        if not compile_success:
            print("❌ Lean code does not compile")
            problem_log["status"] = "failed"
            problem_log["error"] = "Compilation failed"
            problem_log["end_time"] = datetime.now().isoformat()
            problem_log["total_duration_seconds"] = round(time.time() - start_time, 2)
            return False, lean_code, problem_log

        print("✅ Lean code compiles successfully")

        # Check if code has parsable steps
        print("\nChecking for parsable proof steps...")
        parsable, step_count = self._check_parsable_steps(lean_code)

        problem_log["parsability"] = {"is_parsable": parsable, "step_count": step_count}

        if not parsable:
            print("❌ Code does not have properly formatted proof steps")
            problem_log["status"] = "failed"
            problem_log["error"] = "Not parsable"
            problem_log["end_time"] = datetime.now().isoformat()
            problem_log["total_duration_seconds"] = round(time.time() - start_time, 2)
            return False, lean_code, problem_log

        print(f"✅ Code has {step_count} parsable proof steps")

        # Success!
        problem_log["status"] = "success"
        problem_log["end_time"] = datetime.now().isoformat()
        problem_log["total_duration_seconds"] = round(time.time() - start_time, 2)

        return True, lean_code, problem_log

    def _create_user_prompt(self, problem: Dict) -> str:
        """Create user prompt from problem data"""
        problem_text = problem.get("problem", "")
        solution = problem.get("solution", "")

        prompt = f"Problem:\n{problem_text}\n\n"

        if solution:
            prompt += f"Solution/Hint:\n{solution}\n\n"

        prompt += "Please translate this into compilable Lean 4 code with parsable proof steps."

        return prompt

    def _call_openai_api(self, user_prompt: str) -> Optional[str]:
        """Call OpenAI API with retry logic"""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                return response.choices[0].message.content

            except Exception as e:
                print(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    print("Max retries reached")
                    return None

        return None

    def _extract_lean_code(self, response: str) -> str:
        """Extract Lean code from markdown code blocks if present"""
        # Try to find code block with lean or no language specified
        patterns = [r"```lean\n(.*?)```", r"```\n(.*?)```"]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()

        # If no code block found, return the whole response
        return response.strip()

    def _test_lean_compilation(self, lean_code: str) -> Tuple[bool, str]:
        """
        Test if Lean code compiles using lean-check.py

        Returns:
            Tuple of (success, output)
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(lean_code)
            temp_file = f.name

        try:
            result = subprocess.run(
                [sys.executable, self.lean_check_script, "--file", temp_file],
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
            )

            success = result.returncode == 0
            output = result.stdout + result.stderr

            return success, output

        except subprocess.TimeoutExpired:
            return False, "Compilation timed out after 60 seconds"
        except Exception as e:
            return False, f"Error running lean-check: {e}"
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass

    def _check_parsable_steps(self, lean_code: str) -> Tuple[bool, int]:
        """
        Check if Lean code has parsable proof steps

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

    def save_result(
        self, problem_id: str, lean_code: Optional[str], problem_log: Dict
    ) -> str:
        """
        Save result (success or failure) to run directory

        Returns:
            Path to saved file
        """
        # Save individual problem JSON with full log
        json_file = self.run_dir / f"{problem_id}.json"
        with open(json_file, "w") as f:
            json.dump(problem_log, f, indent=2)

        # Also save just the Lean code if it exists
        if lean_code:
            lean_file = self.run_dir / f"{problem_id}.lean"
            with open(lean_file, "w") as f:
                f.write(lean_code)
            print(f"\n✅ Saved results to:")
            print(f"   - {json_file}")
            print(f"   - {lean_file}")
        else:
            print(f"\n📄 Saved log to:")
            print(f"   - {json_file}")

        return str(json_file)

    def process_json_file(
        self, input_file: str, max_problems: Optional[int] = None
    ) -> Dict:
        """
        Process all problems in a JSON file

        Args:
            input_file: Path to input JSON file
            max_problems: Maximum number of problems to process (None = all)

        Returns:
            Dictionary with summary statistics
        """
        print(f"Loading problems from: {input_file}")

        # Update run metadata with input file info
        self.run_metadata["input_file"] = input_file
        self.run_metadata["max_problems"] = max_problems
        self.run_metadata["problem_files"] = []  # Track problem file references

        with open(input_file, "r") as f:
            problems = json.load(f)

        if not isinstance(problems, list):
            problems = [problems]

        if max_problems:
            problems = problems[:max_problems]

        print(f"Processing {len(problems)} problem(s)...")
        print(f"📁 Run directory: {self.run_dir}\n")

        stats = {
            "total": len(problems),
            "successful": 0,
            "failed_compilation": 0,
            "failed_parsable": 0,
            "failed_api": 0,
        }

        for i, problem in enumerate(problems, 1):
            print(f"\n\n{'#'*80}")
            print(f"Problem {i}/{len(problems)}")
            print(f"{'#'*80}")

            success, lean_code, problem_log = self.process_problem(problem)
            problem_id = problem.get("problem_id", f"problem_{i}")

            # Save every problem (success or failure)
            self.save_result(problem_id, lean_code, problem_log)

            # Add reference to problem file in run metadata
            self.run_metadata["problem_files"].append(
                {
                    "problem_id": problem_id,
                    "status": problem_log.get("status"),
                    "file": f"{problem_id}.json",
                }
            )

            # Update stats
            if success:
                stats["successful"] += 1
            else:
                error = problem_log.get("error", "unknown")
                if "Compilation failed" in error:
                    stats["failed_compilation"] += 1
                elif "Not parsable" in error:
                    stats["failed_parsable"] += 1
                else:
                    stats["failed_api"] += 1

        # Complete run metadata
        self.run_metadata["end_time"] = datetime.now().isoformat()
        self.run_metadata["stats"] = stats

        # Print summary
        print(f"\n\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Total problems: {stats['total']}")
        print(f"✅ Successful: {stats['successful']}")
        print(f"❌ Failed (API): {stats['failed_api']}")
        print(f"❌ Failed (Compilation): {stats['failed_compilation']}")
        print(f"❌ Failed (Not parsable): {stats['failed_parsable']}")
        print(f"{'='*80}\n")

        # Save comprehensive run summary
        run_summary_file = self.run_dir / "run_summary.json"
        with open(run_summary_file, "w") as f:
            json.dump(self.run_metadata, f, indent=2)
        print(f"📊 Run summary saved to: {run_summary_file}")
        print(f"📁 All results in: {self.run_dir}\n")

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Process OPC JSON through OpenAI API and test Lean code"
    )
    parser.add_argument("input_file", help="Path to input JSON file with problems")
    parser.add_argument(
        "--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--model", default="gpt-4", help="OpenAI model to use (default: gpt-4)"
    )
    parser.add_argument("--system-prompt", help="Path to system prompt file (optional)")
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="Directory to save successful outputs (default: dataset)",
    )
    parser.add_argument(
        "--lean-check",
        default="scripts/data/lean-check.py",
        help="Path to lean-check.py script (default: scripts/data/lean-check.py)",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        help="Maximum number of problems to process (default: all)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for OpenAI API (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="Maximum tokens for OpenAI API response (default: 4000)",
    )

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    # Check if lean-check script exists
    if not os.path.exists(args.lean_check):
        print(f"Error: lean-check.py script not found: {args.lean_check}")
        print("Make sure you're running from the project root directory")
        sys.exit(1)

    try:
        processor = LeanCodeProcessor(
            api_key=args.api_key,
            model=args.model,
            system_prompt_file=args.system_prompt,
            lean_check_script=args.lean_check,
            dataset_dir=args.dataset_dir,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        processor.process_json_file(args.input_file, max_problems=args.max_problems)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
