"""
OpenAI API client and system prompt management.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import glob
import tempfile
import subprocess
import sys

from openai import OpenAI

from .config import (
    DEFAULT_SYSTEM_PROMPT, 
    MAX_RETRIES, 
    RETRY_DELAY,
)
from .leanspace_manager import LeanWorkspaceManager

# Add the project root to the path to import from src/
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cpe_ai.rag.hf_vector_store import SentenceMxbaiRetriever


class OpenAIClient:
    """Client for interacting with OpenAI API."""

    def __init__(self, api_key: str, model: str, reasoning_effort: str = "high"):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model ID to use
            reasoning_effort: Reasoning effort level (low, medium, high)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.reasoning_effort = reasoning_effort
        
        # Initialize workspace manager
        self.workspace_manager: Optional[LeanWorkspaceManager] = None

        # Initialize local RAG retriever
        print("[INFO] Initializing local RAG retriever...")
        self.retriever = SentenceMxbaiRetriever()
        print("[INFO] Local RAG retriever initialized successfully")

        # Setup iterative tools
        self.tools = []
        self._init_tools()

    def _search_documentation(self, query: str, k: int = 50, n: int = 5) -> Dict[str, any]:
        """
        Search local vector stores for relevant documentation.
        
        Args:
            query: Search query
            k: Number of initial documents to retrieve before reranking
            n: Number of top documents to return after reranking
            
        Returns:
            Dictionary containing search results
        """
        results = {
            "lean_api": [],
            "lean_info": []
        }
        
        # Search lean-api collection
        try:
            api_result = self.retriever.retrieve(
                query=query,
                collection_name="lean_api",
                k=k,
                n=n
            )
            docs = api_result.get("documents", [])
            results["lean_api"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata if hasattr(doc, "metadata") else {}
                }
                for doc in docs
            ]
        except Exception as e:
            print(f"[WARNING] Error searching lean_api: {e}")
        
        # Search lean-info collection  
        try:
            info_result = self.retriever.retrieve(
                query=query,
                collection_name="lean_info",
                k=k,
                n=n
            )
            docs = info_result.get("documents", [])
            results["lean_info"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata if hasattr(doc, "metadata") else {}
                }
                for doc in docs
            ]
        except Exception as e:
            print(f"[WARNING] Error searching lean_info: {e}")
        
        return results

    def _init_tools(self):
        """
        Initialize tools for the ITTER version.

        Tools:
        - search_documentation: Search the local Lean Mathlib documentation and Info vector stores for relevant information.
        - apply_patch: Write changes to Lean files in the workspace and test results of changes.
        - read_file_state: Read the current contents of files in the Lean project workspace.
        """
        # Add local documentation search tool
        self.tools.append({
            "type": "function",
            "function": {
                "name": "search_documentation",
                "description": "Search the Lean Mathlib API documentation and Info documentation for relevant information about Lean syntax, tactics, theorems, and libraries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant documentation."
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of initial documents to retrieve before reranking (default: 50).",
                            "default": 50
                        },
                        "n": {
                            "type": "integer",
                            "description": "Number of top documents to return after reranking (default: 5).",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        })

        # Add Lean code patch tool
        self.tools.append({
            "type": "function",
            "function": {
                "name": "apply_patch",
                "description": "Write changes to lean files in workspace and test results of changes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type":"string"},
                                    "content": {"type":"string"}
                                },
                                "required": ["path","content"]
                            },
                            "description": "Complete contents for any files to create/replace this step."
                        }
                    },
                    "required": ["files"]
                }
            }
        })

        # Add read file state tool
        self.tools.append({
            "type": "function",
            "function": {
                "name": "read_file_state",
                "description": "Read the current contents of files in the Lean project workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of file paths to read."
                        }
                    },
                    "required": ["paths"]
                }
            }
        })

    def call_api(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Call OpenAI API with retry logic.

        Args:
            system_prompt: System prompt for the API
            user_prompt: User prompt with the problem

        Returns:
            Response content or None if failed
        """
        for attempt in range(MAX_RETRIES):
            try:
                # Build API parameters
                api_params = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "reasoning_effort": self.reasoning_effort,
                }

                response = self.client.chat.completions.create(**api_params)

                return response.choices[0].message.content

            except Exception as e:
                print(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    print("Max retries reached")
                    return None

        return None
    
    def call_steps_api(
        self,
        steps_prompt: str,
    ):
        """
        Call OpenAI API with tools and retry logic.

        Args:
            steps_prompt: Prompt for the API

        Returns:
            Response content or None if failed
        """
        response = self.client.chat.completions.create(
            model=self.model,  # or gpt-4o-mini
            messages=[{"role": "user", "content": steps_prompt}],
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)

    def call_itterative_api(
        self,
        patch_prompt: str,
        workspace: "LeanWorkspaceManager",
    ) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Call OpenAI API iteratively with workspace tool calls.
        
        The model can use tools to search files, get workspace state, or apply patches.
        Each iteration, the model makes a tool call and the patch_prompt is regenerated
        to reflect current changes. Exits after NO_COMPILE_LIMIT unsuccessful compile
        attempts or when the model provides a final response.

        Args:
            patch_prompt: Initial prompt for the API
            workspace: LeanWorkspaceManager instance for workspace operations

        Returns:
            Tuple of (final_response, tool_history)
            - final_response: Final model response dict or None if failed
            - tool_history: List of all tool calls and their results
        """
        from .config import NO_COMPILE_LIMIT
        
        # Initialize conversation history with the user's initial request
        messages = [{"role": "user", "content": patch_prompt}]
        tool_history = []
        no_compile_count = 0
        iteration = 0
        max_iterations = 50  # Safety limit to prevent infinite loops
        last_tool_used = None  # Track last tool to prevent consecutive searches

        print(f"[INFO] Starting iterative API call with {len(self.tools)} tools available")

        while iteration < max_iterations:
            iteration += 1
            print(f"\n[ITERATION {iteration}] Calling API...")

            try:
                # Build available tools - disable search_documentation if it was just used
                available_tools = self.tools
                if last_tool_used == "search_documentation":
                    # Filter out search_documentation tool to prevent consecutive searches
                    available_tools = [t for t in self.tools if t["function"]["name"] != "search_documentation"]
                    print("[INFO] Disabling search_documentation (used in last iteration)")
                
                # Make API call with tools
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=available_tools,
                    tool_choice="auto",
                )

                # Get the assistant's response
                response_message = response.choices[0].message
                
                # Add assistant response to conversation history
                messages.append(response_message)

                # Check if model provided a final text response (no more tool calls)
                has_tool_call = False
                tool_calls = response_message.tool_calls

                if tool_calls:
                    has_tool_call = True
                    # Process each tool call
                    for tool_call in tool_calls:
                        # Update last tool used
                        last_tool_used = tool_call.function.name
                        
                        # Handle function tool calls
                        result = self._execute_function_tool(tool_call, workspace)
                        
                        # Track compilation results
                        if tool_call.function.name == "apply_patch":
                            if result.get("compilation", {}).get("success"):
                                no_compile_count = 0  # Reset on success
                                print(f"[SUCCESS] Compilation succeeded!")
                            else:
                                no_compile_count += 1
                                print(f"[FAIL] Compilation failed ({no_compile_count}/{NO_COMPILE_LIMIT})")
                                
                                # Check if we've hit the limit
                                if no_compile_count >= NO_COMPILE_LIMIT:
                                    print(f"[ERROR] Reached NO_COMPILE_LIMIT ({NO_COMPILE_LIMIT})")
                                    return None, tool_history
                        
                        # Log tool call
                        tool_history.append({
                            "iteration": iteration,
                            "tool": tool_call.function.name,
                            "call_id": tool_call.id,
                            "arguments": json.loads(tool_call.function.arguments),
                            "result": result,
                        })

                        # Provide result back to model
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result)
                        })

                # If no tool calls, we're done
                if not has_tool_call:
                    print(f"[INFO] No tool calls in response, iteration complete")
                    return response_message, tool_history

            except Exception as e:
                print(f"[ERROR] API call failed: {e}")
                return None, tool_history

        print(f"[WARN] Reached maximum iterations ({max_iterations})")
        return None, tool_history

    def _execute_function_tool(
        self,
        tool_call,
        workspace: "LeanWorkspaceManager",
    ) -> Dict:
        """
        Execute a function tool call and return the result.

        Args:
            tool_call: Tool call object from ChatCompletion API response
            workspace: LeanWorkspaceManager instance

        Returns:
            Dict containing the tool execution result
        """
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        print(f"[TOOL] Executing {tool_name} with args: {json.dumps(arguments, indent=2)}")

        try:
            if tool_name == "search_documentation":
                # Search local vector stores for documentation
                query = arguments.get("query", "")
                k = arguments.get("k", 50)
                n = arguments.get("n", 5)
                result = self._search_documentation(query=query, k=k, n=n)
                
                # Format result for model
                formatted_result = {
                    "success": True,
                    "query": query,
                    "results": {
                        "lean_api_docs": result.get("lean_api", []),
                        "lean_info_docs": result.get("lean_info", [])
                    },
                    "total_results": len(result.get("lean_api", [])) + len(result.get("lean_info", []))
                }
                return formatted_result
            
            elif tool_name == "apply_patch":
                # Apply file changes and test compilation
                files = arguments.get("files", [])
                result = workspace.apply_patch(files)
                return result

            elif tool_name == "read_file_state":
                # Read file contents
                paths = arguments.get("paths", [])
                result = workspace.read_file_state(paths)
                return result

            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            print(f"[ERROR] Tool execution failed: {e}")
            return {"error": str(e)}


def load_system_prompt(system_prompt_file: Optional[str] = None) -> str:
    """
    Load system prompt from file or use default.

    Args:
        system_prompt_file: Path to system prompt file (optional)

    Returns:
        System prompt string
    """
    if system_prompt_file:
        with open(system_prompt_file, "r") as f:
            return f.read()
    return DEFAULT_SYSTEM_PROMPT