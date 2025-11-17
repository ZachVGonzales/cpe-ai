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
    from .leanspace_manager import LeanWorkspaceManager
    from .config import NO_COMPILE_LIMIT
    from .lean_compiler import (
        extract_lean_code,
        test_lean_compilation,
    )
    from .openai_client import OpenAIClient, load_system_prompt
except ImportError:
    # Fallback for running as direct script
    from leanspace_manager import LeanWorkspaceManager
    from config import NO_COMPILE_LIMIT
    from lean_compiler import (
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
        
        # Initialize OpenAI client
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
            
        # Setup the model's workspace
        print("Setting up model workspace...")
        workspace = LeanWorkspaceManager()

        # Create prompt for step creation
        steps_prompt = self._create_steps_prompt(problem)

        # Call OpenAI API with low reasoning effort to make steps
        steps_start = time.time()

        steps = self.openai_client.call_steps_api(steps_prompt)

        steps_duration = time.time() - steps_start

        problem_log["steps_call"] = {
            "duration_seconds": round(steps_duration, 2),
            "model": self.model,
        }

        # Now call iterative to create proof, save step at each successful compile
        if not steps or "steps" not in steps:
            print("Failed to generate steps from OpenAI API")
            problem_log["status"] = "failed"
            problem_log["error"] = "No steps generated"
            problem_log["end_time"] = datetime.now().isoformat()
            problem_log["total_duration_seconds"] = round(time.time() - start_time, 2)
            return False, None, problem_log

        steps_list = steps.get("steps", [])
        print(f"Generated {len(steps_list)} proof steps")
        
        # Track overall progress
        all_tool_history = []
        last_successful_code = None
        step_logs = []  # Store individual step results for DPO training
        
        for step_info in steps_list:
            step_id = step_info.get("id", "?")
            step_description = step_info.get("description", "")
            print(f"\n{'='*80}")
            print(f"Working on STEP {step_id}: {step_description}")
            print(f"{'='*80}")

            # Create step log entry
            step_log = {
                "step_id": step_id,
                "step_description": step_description,
                "start_time": datetime.now().isoformat(),
                "tool_calls": [],
                "compilation_attempts": [],
                "final_status": "pending",
            }

            # Create patch prompt for this step
            # Get current workspace state for context
            workspace_state = workspace.get_initial_state_for_model()
            
            patch_prompt = self._create_patch_prompt(
                problem,
                compiler_errors=None,
                current_file_state=workspace_state,
                current_goal=step_description,
            )
            
            step_log["patch_prompt"] = patch_prompt

            # Call iterative API to work on this step
            step_start = time.time()
            response, tool_history = self.openai_client.call_itterative_api(
                patch_prompt, workspace
            )
            step_duration = time.time() - step_start
            
            # Add to overall history
            all_tool_history.extend(tool_history)
            step_log["tool_calls"] = tool_history
            step_log["duration_seconds"] = round(step_duration, 2)
            
            print(f"Step {step_id} completed in {step_duration:.2f}s with {len(tool_history)} tool calls")

            if not response:
                print(f"Failed to get response for step {step_id} - hit NO_COMPILE_LIMIT or error")
                step_log["final_status"] = "failed"
                step_log["error"] = "NO_COMPILE_LIMIT reached or API error"
                step_log["end_time"] = datetime.now().isoformat()
                step_logs.append(step_log)
                
                # Save step log immediately
                self._save_step_log(problem_id, step_id, step_log)
                
                problem_log["status"] = "failed"
                problem_log["error"] = f"Failed at step {step_id}: NO_COMPILE_LIMIT reached or API error"
                problem_log["tool_history"] = all_tool_history
                problem_log["step_logs"] = step_logs
                problem_log["end_time"] = datetime.now().isoformat()
                problem_log["total_duration_seconds"] = round(time.time() - start_time, 2)
                return False, last_successful_code, problem_log
            
            # Check for successful compilation in this step's tool history
            step_compiled = False
            for tool_call in tool_history:
                if tool_call.get("tool") == "apply_patch":
                    result = tool_call.get("result", {})
                    compilation = result.get("compilation", {})
                    
                    # Track compilation attempt
                    step_log["compilation_attempts"].append({
                        "success": compilation.get("success", False),
                        "output": compilation.get("output", ""),
                        "files": tool_call.get("arguments", {}).get("files", []),
                    })
                    
                    if compilation.get("success"):
                        # Save the successful code
                        files = tool_call.get("arguments", {}).get("files", [])
                        if files:
                            last_successful_code = "\n\n".join([f["content"] for f in files])
                            step_log["successful_code"] = last_successful_code
                        step_compiled = True
                        print(f"✓ Step {step_id} compiled successfully!")
                        break
            
            # Finalize step log
            step_log["final_status"] = "success" if step_compiled else "failed"
            step_log["end_time"] = datetime.now().isoformat()
            step_logs.append(step_log)
            
            # Save step log immediately for DPO training data
            self._save_step_log(problem_id, step_id, step_log)

        # After all steps, check if we have successful code
        if not last_successful_code:
            print("Failed to generate compilable code for any step")
            problem_log["status"] = "failed"
            problem_log["error"] = "No compilable code generated"
            problem_log["tool_history"] = all_tool_history
            problem_log["step_logs"] = step_logs
            problem_log["end_time"] = datetime.now().isoformat()
            problem_log["total_duration_seconds"] = round(time.time() - start_time, 2)
            return False, None, problem_log
        
        lean_code = last_successful_code
        compile_success = True
        
        # Store final results in problem log
        problem_log["generated_code"] = lean_code
        problem_log["tool_history"] = all_tool_history
        problem_log["step_logs"] = step_logs
        problem_log["compilation"] = {"success": True}

        print("✓ Lean code compiles successfully")

        # Success!
        problem_log["status"] = "success"
        problem_log["end_time"] = datetime.now().isoformat()
        problem_log["total_duration_seconds"] = round(time.time() - start_time, 2)

        return True, lean_code, problem_log
    
    def _create_patch_prompt(
        self,
        problem: Dict[str, Any],
        *,
        compiler_errors: Optional[str] = None,
        current_goal: Optional[str] = None,
        current_file_state: Optional[str] = None,
    ) -> str:
        """
        Create a structured prompt for a code-fixing agent working on Lean 4.

        Parameters
        ----------
        problem : Dict[str, Any]
            Dict with keys:
            - "problem": description of the task/bug/issue
            - "solution": optional hints/partial solution/proof idea
        compiler_errors : Optional[str]
            (Optional) Raw compiler output or error messages to consider.
        current_goal : Optional[str]
            (Optional) The current proof goal / target behavior the code must satisfy.
        current_file_state : Optional[str]
            (Optional) Snapshot of current repository/file contents relevant to the patch
            (e.g., filenames with excerpts or entire files).

        Returns
        -------
        str
            A prompt telling the model to produce minimal patches to meet the current goal,
            including a required "Patch Changes Made" section.
        """
        problem_text = problem.get("problem", "")
        solution = problem.get("solution", "")

        sections = []

        # Task framing & constraints
        header = (
            "You are a Lean 4 code-fixing agent. Your job is to make the **minimal** changes "
            "to existing files necessary to (a) fix the compilation and (b) meet the current goal. "
            "Prefer small, surgical edits over rewrites. Preserve existing style and structure."
        )
        sections.append(header)

        # Problem statement
        sections.append("## Overall Problem\n" + (problem_text or "(none provided)"))

        # Optional hint/solution
        if solution:
            sections.append("## Overall Solution / Hint\n" + solution)

        # Current goal (explicit)
        if current_goal:
            sections.append("## Current Goal\n" + current_goal)

        # Compiler errors (optional)
        if compiler_errors:
            sections.append("## Compiler Errors\n```\n" + compiler_errors.strip() + "\n```")

        # Current file state (optional)
        if current_file_state:
            sections.append("## Current File State\n```\n" + current_file_state.strip() + "\n```")

        # Output requirements for the model
        sections.append(
            "## Requirements\n"
            "- Produce **compilable Lean 4 code** with **complete, working proofs**.\n"
            "- **NEVER use `sorry`, `admit`, or incomplete proofs** - all proofs must be fully completed.\n"
            "- All lemmas and theorems must have **complete tactic proofs** that compile successfully.\n"
            "- Make **minimal patches** to existing files to achieve the current goal.\n"
            "- Do not introduce unrelated refactors.\n"
            "- If a change is necessary across multiple files, keep edits smallest possible.\n"
            "- If you rename/move anything, explain why succinctly.\n"
            "- Use `search_documentation` tool to find the correct Mathlib lemmas and tactics.\n"
            "- Build proofs incrementally - start simple and refine based on compilation errors."
        )
        
        # Final instruction reminder
        sections.append(
            "## Critical Rules\n"
            "- **NO `sorry` statements allowed** - every proof must be complete and compile.\n"
            "- If you don't know how to complete a proof, use `search_documentation` to find relevant Mathlib lemmas.\n"
            "- Stay within the current file structure unless strictly necessary.\n"
            "- Test your changes with `apply_patch` and fix any compilation errors iteratively.\n"
            "- Prioritize working, compilable code that passes `lake build`."
        )

        return "\n\n".join(sections)
    
    def _save_step_log(self, problem_id: str, step_id: int, step_log: Dict[str, Any]) -> None:
        """
        Save individual step log to JSON file for DPO training.
        
        Args:
            problem_id: ID of the problem being processed
            step_id: ID of the step within the problem
            step_log: Dictionary containing step execution details
        """
        try:
            # Create step-specific filename
            step_filename = f"{problem_id}_step_{step_id}.json"
            step_filepath = self.run_dir / step_filename
            
            # Add metadata for DPO training
            dpo_log = {
                "problem_id": problem_id,
                "step_id": step_id,
                "timestamp": datetime.now().isoformat(),
                **step_log,
            }
            
            with open(step_filepath, "w") as f:
                json.dump(dpo_log, f, indent=2)
            
            print(f"[INFO] Saved step log to: {step_filename}")
            
        except Exception as e:
            print(f"[WARN] Could not save step log: {e}")
    
    def _create_steps_prompt(self, problem: dict) -> str:
        problem_text = problem.get("problem", "")
        solution = problem.get("solution", "")

        prompt = (
            "You are a Lean 4 proof planning assistant.\n"
            "Break the given problem into a concise sequence of logical proof steps.\n"
            "Each step should be a short natural-language description of one reasoning action.\n\n"
            "Return a JSON object in this format:\n"
            "{\n"
            '  "steps": [\n'
            '    {"id": 1, "description": "..."},\n'
            '    {"id": 2, "description": "..."}\n'
            "  ]\n"
            "}\n\n"
            f"Problem:\n{problem_text.strip()}\n\n"
        )

        if solution:
            prompt += f"Solution/Hint:\n{solution.strip()}\n\n"

        prompt += "Output only valid JSON — no text or markdown."

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
            print(f"\nSaved log to:")
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
                    if "Compilation failed" in error or "No compilable code" in error:
                        stats["failed_compilation"] += 1
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
        print(f"{'='*80}\n")

        # Save comprehensive run summary
        run_summary_file = self.run_dir / "run_summary.json"
        with open(run_summary_file, "w") as f:
            json.dump(self.run_metadata, f, indent=2)
        print(f"Run summary saved to: {run_summary_file}")
        print(f"All results in: {self.run_dir}\n")

        return stats