
#!/usr/bin/env python3

"""
lean_check.py — compile/run Lean 4 snippets with Lake, flag `sorry`, and (optionally) print axioms.

USAGE EXAMPLES
--------------
# Check a file (compiles only):
python lean_check.py --file path/to/code.lean

# Check an inline snippet:
python lean_check.py --code "def add (a b : Nat) := a + b\n#check add"

# Fail on `sorry` (default) and print axioms of specific decls:
python lean_check.py --file MyProof.lean --axioms myTheorem myLemma

# Run the produced executable if `def main : IO Unit := ...` exists:
python lean_check.py --file Main.lean --run

# Use a persistent working directory (so dependencies aren't re-fetched every time):
python lean_check.py --file My.lean --workdir ./.lean_harness

# Try enabling mathlib (requires git + internet on first run):
python lean_check.py --file UsesMathlib.lean --with-mathlib

REQUIREMENTS
------------
- Lean 4 toolchain installed via `elan` so that `lean` and `lake` are on PATH.
- git (if using --with-mathlib)
- A POSIX shell environment (macOS/Linux). On Windows, run in WSL or Git Bash/PowerShell with compatible tooling.

"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

def run(cmd: List[str], cwd: Path, env=None) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def ensure_tools():
    for tool in ["lean", "lake"]:
        if shutil.which(tool) is None:
            print(f"ERROR: `{tool}` not found on PATH. Please install Lean 4 via elan and ensure `{tool}` is available.", file=sys.stderr)
            sys.exit(2)

def write_lakefile(root: Path, with_mathlib: bool):
    lakefile = root / "lakefile.lean"
    content = [
        "import Lake",
        "open Lake DSL",
        "",
        'package «LeanCheck» where',
        "  -- add configuration options here",
        "",
        'lean_lib «LeanCheck» where',
        "  -- library options",
        "",
        "@[default_target]",
        'lean_exe «Main» where',
        "  root := `Main",
        "",
    ]
    if with_mathlib:
        content.insert(0, 'require mathlib from git "https://github.com/leanprover-community/mathlib4" @ "master"')
        # Note: pin to a specific tag or commit in production for reproducibility.
    lakefile.write_text("\n".join(content))

def write_lean_main(root: Path, user_code: str, axioms: List[str]) -> None:
    src_dir = root / "LeanCheck"
    src_dir.mkdir(parents=True, exist_ok=True)
    main_file = root / "Main.lean"
    # Always treat warnings as errors to catch `sorry` etc.
    prelude = "set_option warningAsError true\n"
    ax_lines = "\n".join([f"#print axioms {name}" for name in axioms])
    main_file.write_text(prelude + user_code.rstrip() + ("\n" + ax_lines if axioms else "") + "\n")

def guess_has_main(user_code: str) -> bool:
    return "def main" in user_code

def parse_args():
    p = argparse.ArgumentParser(description="Compile/run Lean 4 code with Lake; fail on warnings; optionally print axioms.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", type=str, help="Path to a .lean file to compile.")
    g.add_argument("--code", type=str, help="Inline Lean code snippet to compile.")
    p.add_argument("--axioms", nargs="*", default=[], help="Declarations to print axioms for (e.g., myTheorem myLemma).")
    p.add_argument("--run", action="store_true", help="Run the resulting executable if `def main : IO Unit := ...` exists.")
    p.add_argument("--workdir", type=str, default=None, help="Directory to use as the Lake project workspace (created if missing). Defaults to a temp dir.")
    p.add_argument("--with-mathlib", action="store_true", help="Attempt to include mathlib (requires git/network on first run).")
    return p.parse_args()

def main():
    ensure_tools()
    args = parse_args()

    # Project workspace
    if args.workdir:
        work = Path(args.workdir).resolve()
        work.mkdir(parents=True, exist_ok=True)
        temp_created = False
    else:
        work = Path(tempfile.mkdtemp(prefix="lean_harness_"))
        temp_created = True

    try:
        # Lake skeleton
        write_lakefile(work, with_mathlib=args.with_mathlib)

        # Source code
        if args.file:
            user_code = Path(args.file).read_text()
        else:
            user_code = args.code or ""

        write_lean_main(work, user_code, args.axioms)

        # Initialize / update dependencies if needed
        if args.with_mathlib:
            r = run(["lake", "update"], cwd=work)
            print(r.stdout)
            if r.returncode != 0:
                print("ERROR: `lake update` failed.", file=sys.stderr)
                sys.exit(r.returncode)

        # Build
        r = run(["lake", "build"], cwd=work)
        print(r.stdout)
        if r.returncode != 0:
            print("Build failed.", file=sys.stderr)
            sys.exit(r.returncode)

        # Detect `sorry` (Lean marks them as warnings; we already set warnings-as-errors, but double-check output):
        if "declaration uses 'sorry'" in r.stdout or "declaration uses sorry" in r.stdout:
            print("ERROR: Build contains `sorry`.", file=sys.stderr)
            sys.exit(3)

        # Optionally run produced binary
        if args.run:
            # Only run if user code likely defines main; otherwise it's a no-op with a hint.
            exe = work / ".lake" / "build" / "bin" / "Main"
            if exe.exists():
                rr = run([str(exe)], cwd=work)
                print(rr.stdout)
                if rr.returncode != 0:
                    print("Program exited with non-zero status.", file=sys.stderr)
                    sys.exit(rr.returncode)
            else:
                print("No executable produced (no `def main` found?). Skipping run.")

        # If axioms requested, we already injected `#print axioms` into Main.lean; their output appears in build logs.
        # For convenience, extract and echo them succinctly:
        if args.axioms:
            lines = []
            for line in r.stdout.splitlines():
                if line.strip().startswith("axioms:"):
                    lines.append(line)
            if lines:
                print("\nAXIOMS SUMMARY")
                print("----------------")
                for line in lines:
                    print(line)

        print("SUCCESS: Lean code compiled cleanly.", file=sys.stderr)

    finally:
        if temp_created:
            # Keep temp dir for inspection only on failure
            pass

if __name__ == "__main__":
    main()