#!/usr/bin/env python3
"""
Test script for the iterative API implementation.

This script demonstrates how to use the iterative API to solve a simple Lean proof.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from scripts.itter_pipeline.openai_client import OpenAIClient
from scripts.itter_pipeline.leanspace_manager import LeanWorkspaceManager
from scripts.itter_pipeline.config import get_api_key, MODEL_ID

def test_simple_proof():
    """Test the iterative API with a simple proof."""
    
    print("="*80)
    print("Testing Iterative API Implementation")
    print("="*80)
    
    # Get API key
    api_key = get_api_key()
    
    # Initialize client
    print("\n[1] Initializing OpenAI client...")
    client = OpenAIClient(api_key=api_key, model=MODEL_ID)
    
    # Create workspace
    print("\n[2] Creating Lean workspace...")
    workspace = LeanWorkspaceManager()
    
    # Create a simple test prompt
    patch_prompt = """
You are working on a Lean 4 proof. Your task is to prove that 2 + 2 = 4.

Current file state:
src/Main.lean contains:
```lean
import Mathlib

theorem two_plus_two : 2 + 2 = 4 := by
  sorry
```

Replace the `sorry` with a valid proof. Use the `apply_patch` tool to write the corrected file.
Make minimal changes - just replace the sorry with a proof.
"""
    
    print("\n[3] Calling iterative API...")
    print(f"Initial prompt:\n{patch_prompt}\n")
    
    # Call iterative API
    response, tool_history = client.call_itterative_api(
        patch_prompt,
        workspace
    )
    
    print("\n[4] Results:")
    print("="*80)
    
    if response:
        print("✓ SUCCESS: Model completed the task")
        print(f"  Total iterations: {len(tool_history)}")
        print(f"  Tool calls made: {len(tool_history)}")
        
        # Print tool history
        print("\n[5] Tool Call History:")
        for i, call in enumerate(tool_history, 1):
            print(f"\n  Call #{i} (Iteration {call['iteration']}):")
            print(f"    Tool: {call['tool']}")
            
            if call['tool'] == 'apply_patch':
                result = call['result']
                success = result.get('compilation', {}).get('success', False)
                status = "✓ COMPILED" if success else "✗ FAILED"
                print(f"    Compilation: {status}")
                
                if success:
                    print(f"    Files written: {result.get('files_written', [])}")
                else:
                    errors = result.get('compilation', {}).get('output', '')
                    print(f"    Errors: {errors[:200]}...")
                    
            elif call['tool'] == 'read_file_state':
                result = call['result']
                files_read = list(result.get('files', {}).keys())
                print(f"    Files read: {files_read}")
                
            elif call['tool'] == 'file_search':
                print(f"    Queries: {call.get('queries', [])}")
        
        # Show final code if available
        print("\n[6] Final Workspace State:")
        final_state = workspace.read_file_state(["src/Main.lean"])
        if final_state['success']:
            main_content = final_state['files'].get('src/Main.lean', '')
            print("src/Main.lean:")
            print("-" * 80)
            print(main_content)
            print("-" * 80)
        
        print(f"\n[7] Workspace saved at: {workspace.workspace_root}")
        print("    (Will be cleaned up on exit)")
        
    else:
        print("✗ FAILED: Reached NO_COMPILE_LIMIT or encountered error")
        print(f"  Total tool calls: {len(tool_history)}")
        
        # Show why it failed
        failures = 0
        for call in tool_history:
            if call.get('tool') == 'apply_patch':
                if not call.get('result', {}).get('compilation', {}).get('success'):
                    failures += 1
        
        print(f"  Compilation failures: {failures}")
    
    print("\n" + "="*80)
    print("Test Complete")
    print("="*80)
    
    # Cleanup workspace
    workspace.cleanup(keep_on_success=False)

if __name__ == "__main__":
    try:
        test_simple_proof()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
