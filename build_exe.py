"""
Automated PyInstaller Build Script for SignFlow Studio.
Builds the Windows standalone executable into dist/SignFlowStudio.
"""
import subprocess
import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def build():
    print("==================================================")
    print("  SignFlow Studio - Standalone Windows Exe Builder")
    print("==================================================")
    print(f"Project Directory: {ROOT_DIR}")
    
    spec_path = ROOT_DIR / "SignLanguageStudio.spec"
    if not spec_path.exists():
        print(f"[Error] Spec file not found: {spec_path}")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        str(spec_path)
    ]

    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))

    if result.returncode == 0:
        exe_path = ROOT_DIR / "dist" / "SignFlowStudio" / "SignFlowStudio.exe"
        if exe_path.exists():
            print("\n[SUCCESS] Windows Executable built successfully!")
            print(f"Executable location: {exe_path}")
            print(f"Size: {exe_path.stat().st_size / (1024 * 1024):.2f} MB")
            print("You can now run this standalone executable on any compatible Windows PC.")
        else:
            print("[Warning] Build succeeded but executable was not found at expected path.")
    else:
        print(f"\n[Error] PyInstaller build failed with exit code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    build()
