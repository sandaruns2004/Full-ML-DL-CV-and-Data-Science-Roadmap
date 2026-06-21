import sys
import subprocess
import importlib
import os
import platform

def print_header(title):
    print(f"\n{'=' * 50}")
    print(f"🚀 {title}")
    print(f"{'=' * 50}")

def check_os():
    print_header("System Information")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python Executable: {sys.executable}")

def check_python_version():
    print_header("Python Version Check")
    version = sys.version_info
    print(f"Detected Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ FAILED: Python 3.10+ is strictly required for this roadmap.")
        print("   Please upgrade your Python installation.")
        return False
    else:
        print("✅ PASSED: Python version is 3.10 or higher.")
        return True

def check_git():
    print_header("Git Installation Check")
    try:
        result = subprocess.run(['git', '--version'], capture_output=True, text=True, check=True)
        print(f"✅ PASSED: Git is installed: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("❌ FAILED: Git is NOT installed or not in your system PATH.")
        print("   Please install Git from https://git-scm.com/")
        return False
    except Exception as e:
        print(f"❌ ERROR: Could not verify Git: {e}")
        return False

def check_jupyter():
    print_header("Jupyter Environment Check")
    try:
        # Check if jupyter is accessible via command line
        result = subprocess.run(['jupyter', '--version'], capture_output=True, text=True, check=True)
        print("✅ PASSED: Jupyter is installed and accessible.")
        return True
    except FileNotFoundError:
        print("❌ FAILED: Jupyter command not found.")
        print("   Please install it using: pip install jupyterlab")
        return False
    except Exception as e:
        print(f"⚠️ WARNING: Could not verify Jupyter CLI: {e}")
        # Fallback: check if the package is installed
        try:
            importlib.import_module("notebook")
            print("✅ PASSED: Jupyter notebook package is installed.")
            return True
        except ImportError:
            return False

def check_libraries():
    print_header("Machine Learning Libraries Check")
    libraries = {
        "numpy": ("1.21.0", "Numerical computing"),
        "pandas": ("1.3.0", "Data manipulation"),
        "matplotlib": ("3.4.0", "Visualization"),
        "seaborn": ("0.11.0", "Statistical visualization"),
        "scipy": ("1.7.0", "Scientific computing"),
        "sklearn": ("1.0.0", "Classical Machine Learning (scikit-learn)")
    }

    all_passed = True
    for lib, (min_version, desc) in libraries.items():
        try:
            module = importlib.import_module(lib)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {lib.ljust(12)} (v{version.ljust(8)}) - {desc}")
        except ImportError:
            print(f"❌ {lib.ljust(12)} (MISSING)  - Please run `pip install {lib if lib != 'sklearn' else 'scikit-learn'}`")
            all_passed = False

    return all_passed

def main():
    print("\n" + "*" * 60)
    print("🤖 ML Roadmap Environment Setup Validation Tool 🤖")
    print("*" * 60)
    
    check_os()
    python_ok = check_python_version()
    git_ok = check_git()
    jupyter_ok = check_jupyter()
    libs_ok = check_libraries()

    print_header("Final Validation Report")
    
    if python_ok and git_ok and jupyter_ok and libs_ok:
        print("🎉 SUCCESS: Your environment is perfectly configured!")
        print("You are fully ready to tackle the ML, DL, and DS Roadmap.")
    else:
        print("⚠️ ACTION REQUIRED: Some checks failed.")
        print("Please resolve the ❌ FAILED items above before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()
