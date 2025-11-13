"""
LeanWorkspaceManager - Manages Lean 4 project workspace for iterative development.

This module provides functionality to:
- Create and manage Lean 4 project workspaces
- Apply file patches and test compilation
- Read file states
- Track workspace history and changes
"""

import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .lean_compiler import test_lean_compilation


class LeanWorkspaceManager:
    """
    Manages a Lean 4 workspace for iterative code development.
    
    Provides methods that mirror the OpenAI tool calls:
    - apply_patch: Write changes to files and test compilation
    - read_file_state: Read current file contents
    - get_workspace_state: Get complete workspace state
    """

    def __init__(self, lean_version: str = "leanprover/lean4:v4.25.0-rc2", workspace_root: Optional[Path] = None):
        """
        Initialize the workspace manager.

        Args:
            lean_version: Lean toolchain identifier (default: leanprover/lean4:v4.25.0-rc2 -> match mathlib)
            workspace_root: Optional existing workspace root (creates new if None)
        """
        self.lean_version = lean_version
        self.workspace_root = workspace_root or Path(tempfile.mkdtemp(prefix="itter_lean_workspace_"))
        self.history: List[Dict] = []
        self.lakefile_path = self.workspace_root / "lakefile.lean"
        self.toolchain_path = self.workspace_root / "lean-toolchain"
        self.src_dir = self.workspace_root / "src"
        self.main_file = self.src_dir / "Main.lean"
        
        # Initialize workspace if not already set up
        if not self.lakefile_path.exists():
            self._initialize_workspace()
        
        print(f"[INFO] LeanWorkspaceManager initialized at {self.workspace_root}")

    def _initialize_workspace(self) -> None:
        """
        Initialize a new Lean 4 project workspace.
        
        Creates directory structure, lakefile, toolchain, and starter files.
        """
        print(f"[INFO] Initializing new Lean workspace at {self.workspace_root}")
        
        # Create source directory
        self.src_dir.mkdir(parents=True, exist_ok=True)

        # Write toolchain file
        self.toolchain_path.write_text(self.lean_version + "\n", encoding="utf-8")

        # Write lakefile with mathlib dependency
        lakefile_content = """import Lake
open Lake DSL

package itter where
  -- add package configuration options here

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib Main where
  srcDir := "src"
"""
        self.lakefile_path.write_text(lakefile_content, encoding="utf-8")

        # Create starter Main.lean
        main_content = """import Mathlib

def hello : String :=
  "Hello from Lean workspace!"

#eval IO.println s!"{hello}"
"""
        self.main_file.write_text(main_content, encoding="utf-8")

        # Run lake update to fetch dependencies, then cache get for prebuilt binaries
        try:
            print("[INFO] Fetching Mathlib dependencies (this may take a moment)...")
            result = subprocess.run(
                ["lake", "update"],
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            if result.returncode == 0:
                print("[INFO] Mathlib dependencies fetched successfully")
            else:
                print(f"[WARN] Lake update had issues: {result.stderr}")
            
            # Download prebuilt Mathlib binaries to avoid compiling from source
            print("[INFO] Downloading prebuilt Mathlib binaries...")
            cache_result = subprocess.run(
                ["lake", "exe", "cache", "get"],
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for downloading
            )
            if cache_result.returncode == 0:
                print("[INFO] Prebuilt Mathlib binaries downloaded successfully")
            else:
                print(f"[WARN] Cache get had issues: {cache_result.stderr}")
                print("[WARN] Building Mathlib from source may take a very long time")
                
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"[WARN] Could not run 'lake update' or 'cache get': {e}")
            print("[INFO] You may need to run 'lake update && lake exe cache get' manually in the workspace")

        # Log initialization
        self._log_action("initialize_workspace", {
            "lean_version": self.lean_version,
            "workspace_root": str(self.workspace_root),
            "files_created": [
                str(self.lakefile_path),
                str(self.toolchain_path),
                str(self.main_file),
            ]
        })

    def apply_patch(self, files: List[Dict[str, str]]) -> Dict[str, any]:
        """
        Apply file changes to the workspace and test compilation.
        
        This method mirrors the OpenAI tool call 'apply_patch'.

        Args:
            files: List of dicts with 'path' and 'content' keys
                   Example: [{"path": "src/Main.lean", "content": "..."}]

        Returns:
            Dict with:
                - success (bool): Whether all files were written and compiled
                - files_written (List[str]): Paths of files successfully written
                - compilation (Dict): Compilation results
                - errors (List[str]): Any errors encountered
        """
        print(f"\n[APPLY_PATCH] Applying changes to {len(files)} file(s)")
        
        result = {
            "success": False,
            "files_written": [],
            "compilation": {},
            "errors": [],
            "timestamp": datetime.now().isoformat(),
        }

        # Write files
        for file_spec in files:
            path = file_spec.get("path")
            content = file_spec.get("content")
            
            if not path or content is None:
                error_msg = f"Invalid file spec (missing path or content): {file_spec}"
                print(f"[ERROR] {error_msg}")
                result["errors"].append(error_msg)
                continue

            try:
                file_path = self.workspace_root / path
                
                # Create parent directories if needed
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write content
                file_path.write_text(content, encoding="utf-8")
                result["files_written"].append(str(path))
                print(f"[INFO] Wrote {len(content)} characters to {path}")
                
            except Exception as e:
                error_msg = f"Failed to write {path}: {e}"
                print(f"[ERROR] {error_msg}")
                result["errors"].append(error_msg)

        if not result["files_written"]:
            result["errors"].append("No files were successfully written")
            self._log_action("apply_patch", result)
            return result

        # Test compilation
        print("[INFO] Testing compilation...")
        try:
            # For now, compile the main file or first written file
            # In a more sophisticated setup, you might compile specific targets
            compile_success, compile_output = self._run_lake_build()
            
            result["compilation"] = {
                "success": compile_success,
                "output": compile_output,
            }
            
            if compile_success:
                print("[SUCCESS] ✓ Compilation successful")
                result["success"] = True
            else:
                print("[FAILURE] ✗ Compilation failed")
                result["errors"].append("Compilation failed")
                
        except Exception as e:
            error_msg = f"Compilation error: {e}"
            print(f"[ERROR] {error_msg}")
            result["errors"].append(error_msg)
            result["compilation"] = {"success": False, "output": str(e)}

        # Log the action
        self._log_action("apply_patch", result)
        
        return result

    def read_file_state(self, paths: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Read the current contents of files in the workspace.
        
        This method mirrors the OpenAI tool call 'read_file_state'.

        Args:
            paths: List of file paths to read (relative to workspace root)
                   If None, reads all .lean files in the workspace

        Returns:
            Dict with:
                - success (bool): Whether operation succeeded
                - files (Dict[str, str]): Map of path -> content
                - errors (List[str]): Any errors encountered
        """
        if paths is None:
            # Read all .lean files
            paths = [str(p.relative_to(self.workspace_root)) 
                    for p in self.workspace_root.rglob("*.lean")]
            print(f"[READ_FILE_STATE] Reading all .lean files ({len(paths)} found)")
        else:
            print(f"[READ_FILE_STATE] Reading {len(paths)} specified file(s)")

        result = {
            "success": False,
            "files": {},
            "errors": [],
            "timestamp": datetime.now().isoformat(),
        }

        for path in paths:
            try:
                file_path = self.workspace_root / path
                
                if not file_path.exists():
                    error_msg = f"File not found: {path}"
                    print(f"[WARN] {error_msg}")
                    result["errors"].append(error_msg)
                    continue
                
                content = file_path.read_text(encoding="utf-8")
                result["files"][str(path)] = content
                print(f"[INFO] Read {len(content)} characters from {path}")
                
            except Exception as e:
                error_msg = f"Failed to read {path}: {e}"
                print(f"[ERROR] {error_msg}")
                result["errors"].append(error_msg)

        result["success"] = len(result["files"]) > 0
        
        # Log the action
        self._log_action("read_file_state", {
            "paths_requested": paths,
            "files_read": len(result["files"]),
            "errors": result["errors"],
        })
        
        return result

    def get_workspace_state(self) -> Dict[str, any]:
        """
        Get complete workspace state including all files and metadata.

        Returns:
            Dict with:
                - workspace_root (str): Path to workspace
                - files (Dict[str, str]): All .lean files and their contents
                - structure (List[str]): Directory structure
                - history_length (int): Number of actions in history
                - last_compilation (Dict): Last compilation result if any
        """
        print("[GET_WORKSPACE_STATE] Gathering complete workspace state")
        
        # Get only .lean files under src/
        src_lean_files = [str(p.relative_to(self.workspace_root)) 
                          for p in self.src_dir.rglob("*.lean")] if self.src_dir.exists() else []
        file_state = self.read_file_state(paths=src_lean_files)
        
        # Also include lakefile.lean and lean-toolchain if they exist
        config_files = []
        if self.lakefile_path.exists():
            config_files.append("lakefile.lean")
        if self.toolchain_path.exists():
            config_files.append("lean-toolchain")
        
        if config_files:
            config_state = self.read_file_state(paths=config_files)
            # Merge config files into file_state
            file_state["files"].update(config_state.get("files", {}))
        
        # Get directory structure (only under src/)
        structure = []
        if self.src_dir.exists():
            for path in self.src_dir.rglob("*"):
                if path.is_file():
                    rel_path = path.relative_to(self.workspace_root)
                    structure.append(str(rel_path))
        
        # Add config files to structure
        for config_file in config_files:
            if (self.workspace_root / config_file).exists():
                structure.append(config_file)
        
        # Find last compilation result in history
        last_compilation = None
        for action in reversed(self.history):
            if action.get("action") == "apply_patch" and "compilation" in action.get("details", {}):
                last_compilation = action["details"]["compilation"]
                break
        
        state = {
            "workspace_root": str(self.workspace_root),
            "lean_version": self.lean_version,
            "files": file_state.get("files", {}),
            "structure": structure,
            "history_length": len(self.history),
            "last_compilation": last_compilation,
            "timestamp": datetime.now().isoformat(),
        }
        
        return state

    def get_initial_state_for_model(self) -> str:
        """
        Get formatted initial workspace state for the OpenAI model.
        
        This should be called when starting a new conversation to inform
        the model about the current workspace state.

        Returns:
            Formatted string describing workspace state
        """
        state = self.get_workspace_state()
        
        message = f"""# Lean Workspace Initial State

**Workspace Root:** `{state['workspace_root']}`
**Lean Version:** {state['lean_version']}

## Current Files:
"""
        
        for path, content in state['files'].items():
            line_count = content.count('\n') + 1
            message += f"\n### {path} ({line_count} lines)\n"
            message += f"```lean\n{content}\n```\n"
        
        if state['last_compilation']:
            comp = state['last_compilation']
            status = "✓ SUCCESS" if comp.get('success') else "✗ FAILED"
            message += f"\n## Last Compilation: {status}\n"
            if comp.get('output'):
                message += f"```\n{comp['output']}\n```\n"
        
        message += f"\n---\n*History: {state['history_length']} action(s)*\n"
        
        return message

    def _run_lake_build(self) -> Tuple[bool, str]:
        """
        Run lake build in the workspace.

        Returns:
            Tuple of (success, output)
        """
        try:
            result = subprocess.run(
                ["lake", "build"],
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            return success, output
            
        except subprocess.TimeoutExpired:
            return False, "Compilation timed out after 60 seconds"
        except Exception as e:
            return False, f"Error running lake build: {e}"

    def _log_action(self, action: str, details: Dict) -> None:
        """
        Log an action to the workspace history.

        Args:
            action: Name of the action
            details: Details about the action
        """
        log_entry = {
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "details": details,
        }
        self.history.append(log_entry)

    def save_history(self, output_path: Optional[Path] = None) -> Path:
        """
        Save workspace history to a JSON file.

        Args:
            output_path: Where to save (defaults to workspace_root/history.json)

        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = self.workspace_root / "workspace_history.json"
        
        with open(output_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        print(f"[INFO] Saved workspace history to {output_path}")
        return output_path

    def cleanup(self, keep_on_success: bool = True) -> None:
        """
        Clean up the workspace directory.

        Args:
            keep_on_success: If True, only cleanup if last compilation failed
        """
        if keep_on_success:
            # Check last compilation result
            for action in reversed(self.history):
                if action.get("action") == "apply_patch":
                    compilation = action.get("details", {}).get("compilation", {})
                    if compilation.get("success"):
                        print(f"[INFO] Keeping workspace (last compilation succeeded): {self.workspace_root}")
                        return
                    break
        
        try:
            shutil.rmtree(self.workspace_root)
            print(f"[INFO] Cleaned up workspace: {self.workspace_root}")
        except Exception as e:
            print(f"[WARN] Could not clean up workspace: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        # Save history before cleanup
        try:
            self.save_history()
        except Exception as e:
            print(f"[WARN] Could not save history: {e}")
        
        # Cleanup based on whether there was an exception
        self.cleanup(keep_on_success=(exc_type is None))
        return False
