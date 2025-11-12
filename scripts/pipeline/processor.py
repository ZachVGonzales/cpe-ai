"""
Problem processing pipeline for converting math problems to Lean code.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import time
import sys

# Add parent directory to path for importing cpe_ai package
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

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

# Import RAG service
from cpe_ai.services.rag_service import RAGService
from cpe_ai.config.settings import OPENAI_API_KEY, OPENAI_MODEL


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
        use_rag: bool = True,
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
            use_rag: Whether to use RAG service for retrieval-augmented generation
        """
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.skip_complex_geometry = skip_complex_geometry
        self.use_rag = use_rag
        
        # Initialize RAG service if enabled, otherwise use direct OpenAI client
        if use_rag:
            print("🔍 Initializing RAG service with vector store retrieval...")
            # Use the Jinja template from system_prompt_file if provided
            template_path = system_prompt_file if system_prompt_file else None
            self.rag_service = RAGService(
                openai_api_key=api_key,
                openai_model=model,
                prompt_template_path=template_path
            )
            self.system_prompt = None  # RAG service handles prompting
        else:
            print("📝 Using direct OpenAI API without RAG...")
            self.openai_client = OpenAIClient(api_key, model, reasoning_effort)
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
            "use_rag": use_rag,
        }

        print(f"Run directory: {self.run_dir}")

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
                print(f"Skipping problem: {reason}")
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

        # Call OpenAI API (with or without RAG)
        api_start = time.time()
        
        if self.use_rag:
            print("Calling OpenAI API with RAG retrieval...")
            try:
                rag_result = self.rag_service.query(
                    query=user_prompt,
                )
                response = rag_result.get("response")
                problem_log["rag_info"] = {
                    "context_sources": rag_result.get("context_sources", []),
                    "usage": rag_result.get("usage", {}),
                }
            except Exception as e:
                print(f"RAG query failed: {e}")
                response = None
                problem_log["rag_error"] = str(e)
        else:
            print("Calling OpenAI API without RAG...")
            response = self.openai_client.call_api(self.system_prompt, user_prompt)
        
        api_duration = time.time() - api_start

        problem_log["api_call"] = {
            "duration_seconds": round(api_duration, 2),
            "model": self.model,
            "use_rag": self.use_rag,
        }

        if not response:
            print("Failed to get response from OpenAI API")
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
            print("Lean code does not compile")
            problem_log["status"] = "failed"
            problem_log["error"] = "Compilation failed"
            problem_log["end_time"] = datetime.now().isoformat()
            problem_log["total_duration_seconds"] = round(time.time() - start_time, 2)
            return False, lean_code, problem_log

        print("Lean code compiles successfully")

        # Check if code has parsable steps
        print("\nChecking for parsable proof steps...")
        parsable, step_count = check_parsable_steps(lean_code)

        problem_log["parsability"] = {"is_parsable": parsable, "step_count": step_count}

        if not parsable:
            print("Code does not have properly formatted proof steps")
            problem_log["status"] = "failed"
            problem_log["error"] = "Not parsable"
            problem_log["end_time"] = datetime.now().isoformat()
            problem_log["total_duration_seconds"] = round(time.time() - start_time, 2)
            return False, lean_code, problem_log

        print(f"Code has {step_count} parsable proof steps")

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
            print(f"\nSaved results to:")
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
        print(f"Run directory: {self.run_dir}\n")

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
        print(f"Successful: {stats['successful']}")
        print(f"Skipped: {stats['skipped']}")
        print(f"Failed (API): {stats['failed_api']}")
        print(f"Failed (Compilation): {stats['failed_compilation']}")
        print(f"Failed (Not parsable): {stats['failed_parsable']}")
        print(f"{'='*80}\n")

        # Save comprehensive run summary
        run_summary_file = self.run_dir / "run_summary.json"
        with open(run_summary_file, "w") as f:
            json.dump(self.run_metadata, f, indent=2)
        print(f"Run summary saved to: {run_summary_file}")
        print(f"All results in: {self.run_dir}\n")

        return stats

    def process_json_file_batch(
        self, input_file: str, max_problems: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process all problems in a JSON file using OpenAI Batch API.

        Args:
            input_file: Path to input JSON file
            max_problems: Maximum number of problems to process (None = all)

        Returns:
            Dictionary with summary statistics
        """
        print(f"Loading problems from: {input_file}")
        print(f"🔄 Using BATCH API mode\n")

        # Update run metadata with input file info
        self.run_metadata["input_file"] = input_file
        self.run_metadata["max_problems"] = max_problems
        self.run_metadata["batch_mode"] = True
        self.run_metadata["problem_files"] = []

        with open(input_file, "r") as f:
            problems = json.load(f)

        if not isinstance(problems, list):
            problems = [problems]

        if max_problems:
            problems = problems[:max_problems]

        # Filter out problems that should be skipped
        filtered_problems = []
        skipped_count = 0

        for i, problem in enumerate(problems, 1):
            problem_id = problem.get("problem_id", f"problem_{i}")
            problem_text = problem.get("problem", "")

            if self.skip_complex_geometry:
                skip, reason = self._should_skip_problem(problem_text)
                if skip:
                    print(f"Skipping {problem_id}: {reason}")
                    skipped_count += 1
                    # Save skip record
                    skip_log = {
                        "problem_id": problem_id,
                        "problem": problem_text,
                        "status": "skipped",
                        "skip_reason": reason,
                        "timestamp": datetime.now().isoformat(),
                    }
                    self.save_result(problem_id, None, skip_log)
                    continue

            filtered_problems.append((problem_id, problem))

        print(f"\n📦 Preparing batch for {len(filtered_problems)} problem(s)...")

        # Create batch requests with unique custom_ids
        batch_requests = []
        custom_id_counter = {}  # Track duplicates

        for problem_id, problem in filtered_problems:
            # Make custom_id unique by appending counter for duplicates
            if problem_id in custom_id_counter:
                custom_id_counter[problem_id] += 1
                unique_custom_id = f"{problem_id}_{custom_id_counter[problem_id]}"
            else:
                custom_id_counter[problem_id] = 0
                unique_custom_id = problem_id

            user_prompt = self._create_user_prompt(problem)
            batch_requests.append(
                {
                    "custom_id": unique_custom_id,
                    "original_problem_id": problem_id,  # Keep original for reference
                    "system_prompt": self.system_prompt,
                    "user_prompt": user_prompt,
                }
            )

        # Submit batch
        batch_dir = self.run_dir / "batch"
        batch_id, input_file_path = self.openai_client.create_batch_request(
            batch_requests, batch_dir
        )

        if not batch_id:
            print(f"Failed to create batch: {input_file_path}")
            return {"error": "batch_creation_failed"}

        # Save batch info
        batch_info = {
            "batch_id": batch_id,
            "input_file": input_file_path,
            "created_at": datetime.now().isoformat(),
            "problem_count": len(filtered_problems),
        }

        batch_info_file = batch_dir / "batch_info.json"
        with open(batch_info_file, "w") as f:
            json.dump(batch_info, f, indent=2)

        print(f"Batch info saved to: {batch_info_file}")
        print(f"\nWaiting for batch to complete (this may take a while)...")

        # Wait for batch to complete
        final_status = self.openai_client.wait_for_batch(batch_id)

        if final_status != "completed":
            print(f"Batch did not complete successfully: {final_status}")
            return {"error": f"batch_{final_status}"}

        # Retrieve results
        output_file = batch_dir / f"batch_output_{batch_id}.jsonl"
        results = self.openai_client.retrieve_batch_results(batch_id, output_file)

        if not results:
            print("Failed to retrieve batch results")
            return {"error": "batch_retrieval_failed"}

        print(f"\nProcessing {len(results)} batch results...")

        # Process results - need to map back from unique custom_ids
        stats = {
            "total": len(problems),
            "successful": 0,
            "skipped": skipped_count,
            "failed_compilation": 0,
            "failed_parsable": 0,
            "failed_api": 0,
        }

        # Create mapping from unique_custom_id to problem data
        problem_map = {}
        for problem_id, problem in filtered_problems:
            if problem_id in custom_id_counter and custom_id_counter[problem_id] > 0:
                # This problem_id has duplicates, find the right unique_custom_id
                count = 0
                for pid, prob in filtered_problems:
                    if pid == problem_id:
                        unique_id = pid if count == 0 else f"{pid}_{count}"
                        problem_map[unique_id] = (pid, prob)
                        count += 1
                        if count > custom_id_counter[problem_id]:
                            break
            else:
                problem_map[problem_id] = (problem_id, problem)

        for unique_custom_id, (problem_id, problem) in problem_map.items():
            print(f"\n{'='*80}")
            print(f"Processing result: {unique_custom_id}")
            if unique_custom_id != problem_id:
                print(f"  (Original ID: {problem_id})")
            print(f"{'='*80}")

            response = results.get(unique_custom_id)

            if not response:
                print(f"No response from API for {problem_id}")
                problem_log = {
                    "problem_id": problem_id,
                    "problem": problem.get("problem", ""),
                    "status": "failed",
                    "error": "No response from batch API",
                    "timestamp": datetime.now().isoformat(),
                }
                self.save_result(problem_id, None, problem_log)
                stats["failed_api"] += 1
                continue

            # Extract and test Lean code
            lean_code = extract_lean_code(response)

            # Test compilation
            print("Testing Lean code compilation...")
            project_root = Path(__file__).parent.parent.parent
            compile_success, compile_output = test_lean_compilation(
                lean_code, project_root
            )

            # Check parsability
            parsable, step_count = check_parsable_steps(lean_code)

            # Create problem log
            problem_log = {
                "problem_id": problem_id,
                "problem": problem.get("problem", ""),
                "solution": problem.get("solution", ""),
                "metadata": problem.get("metadata", {}),
                "generated_code": lean_code,
                "compilation": {
                    "success": compile_success,
                    "output": compile_output,
                },
                "parsability": {
                    "is_parsable": parsable,
                    "step_count": step_count,
                },
                "timestamp": datetime.now().isoformat(),
                "batch_mode": True,
            }

            # Determine success
            if compile_success and parsable:
                problem_log["status"] = "success"
                stats["successful"] += 1
                print(f"Success! {step_count} parsable steps")
            elif not compile_success:
                problem_log["status"] = "failed"
                problem_log["error"] = "Compilation failed"
                stats["failed_compilation"] += 1
                print("Compilation failed")
            else:
                problem_log["status"] = "failed"
                problem_log["error"] = "Not parsable"
                stats["failed_parsable"] += 1
                print("Not parsable")

            self.save_result(problem_id, lean_code, problem_log)

        # Complete run metadata
        self.run_metadata["end_time"] = datetime.now().isoformat()
        self.run_metadata["stats"] = stats
        self.run_metadata["batch_id"] = batch_id

        # Print summary
        print(f"\n\n{'='*80}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*80}")
        print(f"Batch ID: {batch_id}")
        print(f"Total problems: {stats['total']}")
        print(f"Successful: {stats['successful']}")
        print(f"Skipped: {stats['skipped']}")
        print(f"Failed (API): {stats['failed_api']}")
        print(f"Failed (Compilation): {stats['failed_compilation']}")
        print(f"Failed (Not parsable): {stats['failed_parsable']}")
        print(f"{'='*80}\n")

        # Save comprehensive run summary
        run_summary_file = self.run_dir / "run_summary.json"
        with open(run_summary_file, "w") as f:
            json.dump(self.run_metadata, f, indent=2)
        print(f"Run summary saved to: {run_summary_file}")
        print(f"All results in: {self.run_dir}\n")

        return stats
