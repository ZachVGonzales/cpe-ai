"""
Problem processing pipeline for converting math problems to Lean code.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import time
import sys

# Handle relative imports when run as script or module
try:
    from .lean_compiler import (
        check_parsable_steps,
        extract_lean_code,
        test_lean_compilation,
    )
    from .openai_client import OpenAIClient, load_system_prompt
except ImportError:
    # Fallback for running as direct script
    from lean_compiler import (
        check_parsable_steps,
        extract_lean_code,
        test_lean_compilation,
    )
    from openai_client import OpenAIClient, load_system_prompt


class LeanCodeProcessor:
    """Process math problems through OpenAI API and test Lean code."""

    def __init__(
        self,
        api_key: str,
        model: str,
        system_prompt_file: Optional[str] = None,
        dataset_dir: str = "dataset",
        reasoning_effort: str = "high",
        skip_complex_geometry: bool = True,
    ):
        """
        Initialize the processor.

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
            system_prompt_file: Path to system prompt file
            dataset_dir: Directory to save successful outputs
            reasoning_effort: Reasoning effort level (low, medium, high)
            skip_complex_geometry: Skip problems with complex geometry
        """
        self.openai_client = OpenAIClient(api_key, model, reasoning_effort)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.skip_complex_geometry = skip_complex_geometry
        self.system_prompt = load_system_prompt(system_prompt_file)

        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(dataset_dir) / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Store run metadata
        self.run_metadata: Dict[str, Any] = {
            "timestamp": timestamp,
            "start_time": datetime.now().isoformat(),
            "model": model,
            "reasoning_effort": reasoning_effort,
            "skip_complex_geometry": skip_complex_geometry,
            "system_prompt_file": system_prompt_file,
        }

        print(f"📁 Run directory: {self.run_dir}")

    def process_problem(
        self, problem: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Process a single problem through OpenAI API and test the result.

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

        # Check if problem should be skipped
        if self.skip_complex_geometry:
            skip, reason = self._should_skip_problem(problem_text)
            if skip:
                print(f"⏭️  Skipping problem: {reason}")
                problem_log["status"] = "skipped"
                problem_log["skip_reason"] = reason
                problem_log["end_time"] = datetime.now().isoformat()
                problem_log["total_duration_seconds"] = round(
                    time.time() - start_time, 2
                )
                return False, None, problem_log

        # Create prompt from problem
        user_prompt = self._create_user_prompt(problem)
        problem_log["user_prompt"] = user_prompt

        # Call OpenAI API
        print("Calling OpenAI API...")
        api_start = time.time()
        response = self.openai_client.call_api(self.system_prompt, user_prompt)
        api_duration = time.time() - api_start

        problem_log["api_call"] = {
            "duration_seconds": round(api_duration, 2),
            "model": self.model,
        }

        if not response:
            print("❌ Failed to get response from OpenAI API")
            problem_log["status"] = "failed"
            problem_log["error"] = "No response from API"
            problem_log["end_time"] = datetime.now().isoformat()
            problem_log["total_duration_seconds"] = round(time.time() - start_time, 2)
            return False, None, problem_log

        # Extract Lean code from response
        lean_code = extract_lean_code(response)
        problem_log["generated_code"] = lean_code

        # Test if code compiles
        print("\nTesting Lean code compilation...")
        compile_start = time.time()
        project_root = Path(__file__).parent.parent.parent
        compile_success, compile_output = test_lean_compilation(lean_code, project_root)
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
        parsable, step_count = check_parsable_steps(lean_code)

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

    def _create_user_prompt(self, problem: Dict[str, Any]) -> str:
        """Create user prompt from problem data."""
        problem_text = problem.get("problem", "")
        solution = problem.get("solution", "")

        prompt = f"Problem:\n{problem_text}\n\n"

        if solution:
            prompt += f"Solution/Hint:\n{solution}\n\n"

        prompt += "Please translate this into compilable Lean 4 code with parsable proof steps."

        return prompt

    def _should_skip_problem(self, problem_text: str) -> Tuple[bool, str]:
        """
        Check if a problem should be skipped based on complexity indicators.

        Returns:
            Tuple of (should_skip, reason)
        """
        problem_lower = problem_text.lower()

        # Indicators of complex geometry that might not be suitable for Lean
        complex_geometry_indicators = [
            ("circumcenter", "complex geometric construction (circumcenter)"),
            ("orthocenter", "complex geometric construction (orthocenter)"),
            ("incircle", "complex geometric construction (incircle)"),
            ("excircle", "complex geometric construction (excircle)"),
            ("simson line", "complex geometric construction (Simson line)"),
            ("nine-point circle", "complex geometric construction (nine-point circle)"),
            ("construct", "geometric construction required"),
            ("draw a line", "geometric construction required"),
            ("take points", "complex point construction"),
        ]

        for indicator, reason in complex_geometry_indicators:
            if indicator in problem_lower:
                return True, reason

        # Check for multiple geometric points/constructions (heuristic)
        if problem_lower.count("point") > 8 or problem_lower.count("line") > 6:
            return True, "too many geometric elements (likely complex construction)"

        return False, ""

    def save_result(
        self, problem_id: str, lean_code: Optional[str], problem_log: Dict[str, Any]
    ) -> str:
        """
        Save result (success or failure) to run directory.

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
    ) -> Dict[str, Any]:
        """
        Process all problems in a JSON file.

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
            "skipped": 0,
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
                status = problem_log.get("status")
                if status == "skipped":
                    stats["skipped"] += 1
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
        print(f"⏭️  Skipped: {stats['skipped']}")
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
