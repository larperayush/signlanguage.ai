"""
SignFlow Studio - Sign Language Recognition Desktop Application.
Development launcher script.
"""
import sys
import os

# Add root directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from app.main import main

if __name__ == "__main__":
    main()
