# Quick Reference: Iterative API Tool Calling

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenAI Client                            │
│  ┌───────────────────────────────────────────────────┐      │
│  │  call_itterative_api(prompt, workspace)           │      │
│  │                                                   │      │
│  │  1. Initialize conversation with prompt           │      │
│  │  2. Loop until model stops or limit reached:      │      │
│  │     a. Call responses.create() with tools         │      │
│  │     b. Execute function calls                     │      │
│  │     c. Track compilation results                  │      │
│  │     d. Return results to model                    │      │
│  │  3. Return final response + tool history          │      │
│  └───────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              LeanWorkspaceManager                           │
│  ┌───────────────────────────────────────────────────┐      │
│  │  apply_patch(files)                               │      │
│  │    • Write files to disk                          │      │
│  │    • Run `lake build`                             │      │
│  │    • Return compilation result                    │      │
│  │                                                   │      │
│  │  read_file_state(paths)                           │      │
│  │    • Read specified files                         │      │
│  │    • Return contents                              │      │
│  │                                                   │      │
│  │  get_workspace_state()                            │      │
│  │    • Return complete workspace snapshot           │      │
│  └───────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Tool Call Flow

```
User Prompt
    │
    ▼
┌─────────────────────────────┐
│ Iteration 1                 │
│ ┌─────────────────────────┐ │
│ │ Model calls file_search │ │
│ │ → Searches docs         │ │
│ │ → Returns results       │ │
│ └─────────────────────────┘ │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Iteration 2                 │
│ ┌─────────────────────────┐ │
│ │ Model reads file state  │ │
│ │ → read_file_state       │ │
│ │ → Returns contents      │ │
│ └─────────────────────────┘ │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Iteration 3                 │
│ ┌─────────────────────────┐ │
│ │ Model applies patch     │ │
│ │ → apply_patch           │ │
│ │ → Compilation FAILED    │ │
│ │ → no_compile_count = 1  │ │
│ └─────────────────────────┘ │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Iteration 4                 │
│ ┌─────────────────────────┐ │
│ │ Model applies patch     │ │
│ │ → apply_patch           │ │
│ │ → Compilation SUCCESS   │ │
│ │ → no_compile_count = 0  │ │
│ └─────────────────────────┘ │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Iteration 5                 │
│ ┌─────────────────────────┐ │
│ │ Model provides message  │ │
│ │ → No tool calls         │ │
│ │ → DONE                  │ │
│ └─────────────────────────┘ │
└─────────────────────────────┘
```

## API Call Format

### Request (to OpenAI)
```python
response = client.responses.create(
    model="gpt-5",
    input=[
        {"role": "user", "content": "Fix the proof..."},
        # ... previous conversation ...
    ],
    tools=[
        {"type": "file_search", "vector_store_ids": [...]},
        {"type": "function", "function": {...}},
        {"type": "function", "function": {...}},
    ]
)
```

### Response (from OpenAI)
```python
response.output = [
    {
        "type": "reasoning",
        "content": [...],
        "summary": [...]
    },
    {
        "type": "function_call",
        "id": "fc_abc123",
        "call_id": "call_xyz789",
        "name": "apply_patch",
        "arguments": '{"files": [...]}'
    }
]
```

### Feedback (to OpenAI)
```python
input_list.append({
    "type": "function_call_output",
    "call_id": "call_xyz789",
    "output": '{"success": true, "compilation": {...}}'
})
```

## Configuration Quick Ref

| Setting | Location | Default | Purpose |
|---------|----------|---------|---------|
| `MODEL_ID` | config.py | `"gpt-5"` | Which model to use |
| `NO_COMPILE_LIMIT` | config.py | `5` | Max consecutive failures |
| `LEAN_BUILD_TIMEOUT` | config.py | `60` | Compilation timeout (sec) |
| `max_iterations` | openai_client.py | `50` | Safety iteration limit |

## Return Values

### `call_itterative_api()` Returns

```python
response, tool_history = call_itterative_api(prompt, workspace)

# response: Dict or None
# - None if failed (hit NO_COMPILE_LIMIT or error)
# - Dict with model's final output if success

# tool_history: List[Dict]
[
    {
        "iteration": 1,
        "tool": "file_search",
        "call_id": "call_abc",
        "queries": ["query1", "query2"],
        "status": "completed"
    },
    {
        "iteration": 2,
        "tool": "apply_patch",
        "call_id": "call_def",
        "arguments": {"files": [...]},
        "result": {
            "success": True,
            "compilation": {"success": True, "output": "..."},
            "files_written": ["src/Main.lean"]
        }
    }
]
```

## Common Patterns

### Check if compilation succeeded
```python
for call in tool_history:
    if call["tool"] == "apply_patch":
        if call["result"]["compilation"]["success"]:
            print("✓ Compiled!")
            break
```

### Extract final successful code
```python
last_code = None
for call in tool_history:
    if call["tool"] == "apply_patch":
        if call["result"]["compilation"]["success"]:
            files = call["arguments"]["files"]
            last_code = "\n\n".join([f["content"] for f in files])
```

### Count failures
```python
failures = sum(
    1 for call in tool_history
    if call["tool"] == "apply_patch"
    and not call["result"]["compilation"]["success"]
)
```

## Error Codes

| Condition | Return | Reason |
|-----------|--------|--------|
| Success | `(response, history)` | Model completed task |
| Hit limit | `(None, history)` | `no_compile_count >= NO_COMPILE_LIMIT` |
| Max iterations | `(None, history)` | `iteration >= max_iterations` (50) |
| API error | `(None, history)` | Exception during API call |

## Debugging Commands

```python
# Print all tool calls
for i, call in enumerate(tool_history, 1):
    print(f"{i}. {call['tool']} (iter {call['iteration']})")

# Print compilation results
for call in tool_history:
    if call["tool"] == "apply_patch":
        comp = call["result"]["compilation"]
        status = "✓" if comp["success"] else "✗"
        print(f"{status} {comp.get('output', '')[:100]}")

# Show file search queries
for call in tool_history:
    if call["tool"] == "file_search":
        print(f"Searched: {call['queries']}")

# Get final workspace state
final_state = workspace.get_workspace_state()
print(f"Files: {list(final_state['files'].keys())}")
```

## Step-by-Step Usage

```python
# 1. Import
from openai_client import OpenAIClient
from leanspace_manager import LeanWorkspaceManager
from config import get_api_key, MODEL_ID

# 2. Initialize
api_key = get_api_key()
client = OpenAIClient(api_key=api_key, model=MODEL_ID)
workspace = LeanWorkspaceManager()

# 3. Create prompt
prompt = "Fix this Lean proof: ..."

# 4. Call API
response, history = client.call_itterative_api(prompt, workspace)

# 5. Check result
if response:
    print("Success!")
    # Extract code...
else:
    print("Failed")
    # Analyze history...

# 6. Cleanup
workspace.cleanup(keep_on_success=False)
```

## Limits & Timeouts

| Limit | Value | Location |
|-------|-------|----------|
| Max iterations | 50 | `call_itterative_api()` |
| Compile failures | 5 | `NO_COMPILE_LIMIT` |
| Build timeout | 60s | `LEAN_BUILD_TIMEOUT` |
| Lake update timeout | 300s | `LeanWorkspaceManager._initialize_workspace()` |
| Cache get timeout | 600s | `LeanWorkspaceManager._initialize_workspace()` |

## File Paths

All paths relative to workspace root:
- `lakefile.lean` - Lake build configuration
- `lean-toolchain` - Lean version specification
- `src/Main.lean` - Main Lean file
- `src/*.lean` - Additional Lean files

Use relative paths in tool calls:
```python
{"path": "src/Main.lean", "content": "..."}
```

Not absolute paths:
```python
{"path": "/tmp/workspace/src/Main.lean", ...}  # ✗ Wrong
```
