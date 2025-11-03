# Pipeline Module Refactoring

## Overview

The monolithic `process_opc.py` script has been refactored into a modular architecture with a `pipeline/` directory containing separate, focused modules.

## Directory Structure

```
scripts/
├── process_opc.py          # Main entry point (refactored)
└── pipeline/
    ├── __init__.py         # Pipeline package initialization
    ├── config.py           # Configuration & environment setup
    ├── openai_client.py    # OpenAI API interactions
    ├── lean_compiler.py    # Lean code compilation & validation
    └── processor.py        # Main problem processing pipeline
```

## Module Breakdown

### `config.py`

- **Purpose**: Configuration constants and environment setup
- **Contents**:
  - `MODEL_ID`, `TEMPERATURE`: Model constants
  - `MAX_RETRIES`, `RETRY_DELAY`: API retry configuration
  - `LEAN_BUILD_TIMEOUT`: Compilation timeout
  - `DEFAULT_SYSTEM_PROMPT`: Default Lean code generation prompt
  - `setup_environment()`: Initialize dependencies and load .env
  - `get_api_key()`: Retrieve API key from parameter or environment

### `openai_client.py`

- **Purpose**: OpenAI API interactions
- **Contents**:
  - `OpenAIClient` class: Handles API calls with retry logic
  - `load_system_prompt()`: Load system prompt from file or use default

### `lean_compiler.py`

- **Purpose**: Lean code compilation and validation
- **Contents**:
  - `test_lean_compilation()`: Compile code using `lake build`
  - `check_parsable_steps()`: Validate proof step formatting
  - `extract_lean_code()`: Extract Lean code from markdown responses

### `processor.py`

- **Purpose**: Main problem processing pipeline
- **Contents**:
  - `LeanCodeProcessor` class: Orchestrates the entire pipeline
    - `process_problem()`: Process individual problems
    - `process_json_file()`: Batch processing with statistics
    - Helper methods for prompts, skipping rules, and result saving

### `process_opc.py`

- **Purpose**: Command-line entry point
- **Contents**:
  - Argument parsing
  - Environment initialization
  - Instantiation and execution of `LeanCodeProcessor`

## Benefits of This Refactoring

1. **Modularity**: Each module has a single, clear responsibility
2. **Testability**: Individual modules can be tested independently
3. **Reusability**: Components can be imported and used in other projects
4. **Maintainability**: Easier to locate and update specific functionality
5. **Scalability**: Simpler to extend or add new features
6. **Code Organization**: Clear separation of concerns

## Usage

The command-line interface remains the same:

```bash
python scripts/process_opc.py <input_file> [options]
```

All arguments are unchanged from the original implementation.

## Dependencies

The refactored code maintains all original dependencies:

- `openai`: For API interactions
- `python-dotenv`: For environment variable loading (optional but recommended)
- Standard library: `json`, `re`, `subprocess`, `pathlib`, `argparse`, `datetime`, etc.
