#!/usr/bin/env python3
"""
test_setup.py - Verify that the OPC processing setup is working correctly

This script checks:
1. Required Python packages are installed
2. Lean 4 is installed and working
3. OpenAI API key is configured
4. lean-check.py script is accessible
"""

import os
import sys
import subprocess
from pathlib import Path

try:
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()
except ImportError:
    # python-dotenv not installed, continue with system env vars
    pass


def check_python_packages():
    """Check if required Python packages are installed"""
    print("Checking Python packages...")
    required = ["openai", "pandas", "pyarrow", "dotenv"]
    missing = []

    for package in required:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("✅ All required packages installed\n")
    return True


def check_lean_installation():
    """Check if Lean 4 is installed"""
    print("Checking Lean 4 installation...")

    try:
        # Check lean
        result = subprocess.run(
            ["lean", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  ✅ lean: {version}")
        else:
            print("  ❌ lean command failed")
            return False

        # Check lake
        result = subprocess.run(
            ["lake", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  ✅ lake: {version}")
        else:
            print("  ❌ lake command failed")
            return False

        print("✅ Lean 4 is properly installed\n")
        return True

    except FileNotFoundError:
        print("  ❌ lean/lake not found in PATH")
        print("\nInstall Lean 4 with:")
        print(
            "  curl -L https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh"
        )
        print("  (then restart terminal)")
        return False
    except Exception as e:
        print(f"  ❌ Error checking Lean: {e}")
        return False


def check_api_key():
    """Check if OpenAI API key is configured"""
    print("Checking OpenAI API key...")

    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print(f"  ✅ .env file found")
    else:
        print(f"  ⚠️  .env file not found (optional)")
        print(f"     You can copy .env.example to .env and add your API key")

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        # Don't print the actual key, just confirm it exists
        masked = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 11 else "***"
        print(f"  ✅ OPENAI_API_KEY set: {masked}")
        print("✅ API key is configured\n")
        return True
    else:
        print("  ❌ OPENAI_API_KEY environment variable not set")
        print("\nSet your API key with:")
        print("  1. Copy .env.example to .env: cp .env.example .env")
        print("  2. Edit .env and add your API key")
        print("  OR")
        print("  3. Export in terminal: export OPENAI_API_KEY='your-api-key-here'")
        return False


def check_lean_check_script():
    """Check if lean-check.py script exists and is accessible"""
    print("Checking lean-check.py script...")

    script_path = Path("scripts/data/lean-check.py")

    if not script_path.exists():
        print(f"  ❌ Script not found at: {script_path}")
        print("\nMake sure you're running from the project root directory")
        return False

    print(f"  ✅ Found at: {script_path}")

    # Try to run it with --help
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print("  ✅ Script is executable")
            print("✅ lean-check.py is accessible\n")
            return True
        else:
            print("  ❌ Script returned error")
            return False
    except Exception as e:
        print(f"  ❌ Error running script: {e}")
        return False


def check_process_script():
    """Check if process_opc.py script exists"""
    print("Checking process_opc.py script...")

    script_path = Path("scripts/process_opc.py")

    if not script_path.exists():
        print(f"  ❌ Script not found at: {script_path}")
        return False

    print(f"  ✅ Found at: {script_path}")
    print("✅ process_opc.py is available\n")
    return True


def check_data_files():
    """Check if example data files exist"""
    print("Checking for example data files...")

    data_file = Path("data/OPC/generic-OPC.json")

    if not data_file.exists():
        print(f"  ⚠️  Example data not found at: {data_file}")
        print("  (This is optional)")
        return True  # Not critical

    print(f"  ✅ Found example data: {data_file}")
    print("✅ Example data available\n")
    return True


def main():
    print("=" * 80)
    print("OPC Processing Setup Verification")
    print("=" * 80)
    print()

    checks = [
        ("Python Packages", check_python_packages),
        ("Lean 4 Installation", check_lean_installation),
        ("OpenAI API Key", check_api_key),
        ("lean-check.py Script", check_lean_check_script),
        ("process_opc.py Script", check_process_script),
        ("Example Data Files", check_data_files),
    ]

    results = []
    for name, check_func in checks:
        try:
            success = check_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ Unexpected error in {name}: {e}\n")
            results.append((name, False))

    print("=" * 80)
    print("Summary")
    print("=" * 80)

    all_passed = all(success for _, success in results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    print("=" * 80)

    if all_passed:
        print(
            "\n✅ All checks passed! You're ready to use the OPC processing pipeline."
        )
        print("\nNext steps:")
        print("1. Run a test: ./scripts/example_run.sh")
        print(
            "2. Or process problems: python scripts/process_opc.py data/OPC/generic-OPC.json --max-problems 1"
        )
        return 0
    else:
        print(
            "\n❌ Some checks failed. Please address the issues above before proceeding."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
