# cpe-ai

Sustainable LLMs

## Creating Env

### Create Conda Env (Optional)

```bash
conda create -n lean4 python=3.11 -y
conda activate lean4
```

### Install LEAN4

```bash
curl -L https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh
```

- NOTE: restart terminal after this

### Confirm Installation

```bash
lean --version
lake --version
```

## Testing Lean Code

The following lists the steps necessary to test LEAN code (compilation & run)

1. generate a lean code file (see data/OPC/example.lean for example)

2. using the provided test script run the following

```bash
# list all options:
python scripts/data/lean-check.py --help
```

```bash
# build and run:
python scripts/data/lean-check.py --file data/OPC/filename... --run
```

## OPC Processing Pipeline

This project includes a complete pipeline for processing mathematical problems through the OpenAI API, generating Lean 4 code, and validating the results.

### Quick Start

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Set your OpenAI API key:**

```bash
# Option 1: Use .env file (recommended)
cp .env.example .env
# Then edit .env and add your API key

# Option 2: Export as environment variable
export OPENAI_API_KEY='your-api-key-here'
```

3. **Verify setup (optional):**

```bash
python scripts/test_setup.py
```

4. **Run the processing pipeline:**

```bash
# Process a single problem (test)
python scripts/process_opc.py data/OPC/generic-OPC.json --max-problems 1

# Process all problems
python scripts/process_opc.py data/OPC/generic-OPC.json

# Use custom system prompt
python scripts/process_opc.py data/OPC/generic-OPC.json \
  --system-prompt data/OPC/System-Prompt_CHAP1.txt
```

### What It Does

The `process_opc.py` script:

1. ✅ Reads math problems from JSON files
2. ✅ Sends each problem to OpenAI API with a specialized system prompt
3. ✅ Receives Lean 4 code with parsable proof steps
4. ✅ Tests if the code compiles using `lean-check.py`
5. ✅ Validates that proof steps are properly formatted
6. ✅ Saves successful results to a dataset directory

### Output

Each run creates a timestamped directory with comprehensive logging:

```
dataset/
└── 20251030_143022/              # Timestamp: YYYYMMDD_HHMMSS
    ├── run_summary.json          # Run metadata and stats
    ├── BMOSL_2016_14.json        # Problem 1: Full log
    ├── BMOSL_2016_14.lean        # Problem 1: Lean code (if successful)
    ├── BMOSL_2016_15.json        # Problem 2: Full log
    ├── BMOSL_2016_15.lean        # Problem 2: Lean code (if successful)
    └── ...
```

**`run_summary.json`** contains:

- Timestamp and duration
- Model configuration (model, temperature)
- Input file and max_problems setting
- Overall statistics (success/failure counts)
- List of problem files with status

**Each problem JSON file** (`{problem_id}.json`) contains:

- Original problem and solution
- User prompt sent to API
- API call details (duration, model settings)
- Generated Lean code
- Compilation results (success, duration, output)
- Parsability check results (step count)
- Timestamps and total duration
- Status and any errors

### Validation

The script performs two types of validation:

1. **Compilation Check** - Code must build with `lake build` and pass `warningAsError`
2. **Parsability Check** - Code must have properly formatted `[STEP_X: ...]` proof steps

### Example Usage

```bash
# Process 5 problems
python scripts/process_opc.py data/OPC/generic-OPC.json --max-problems 5

# Custom configuration
python scripts/process_opc.py data/OPC/generic-OPC.json \
  --dataset-dir dataset/experiment1 \
  --system-prompt data/OPC/System-Prompt-GENARIC.txt
```

### Full Documentation

See [scripts/README_PROCESS_OPC.md](scripts/README_PROCESS_OPC.md) for:

- Complete command-line options
- Input/output formats
- Troubleshooting guide
- Advanced usage examples
