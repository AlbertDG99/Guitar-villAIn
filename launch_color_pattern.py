#!/usr/bin/env python3
"""
Launcher script for color_pattern_visualizer.py
This script can be run from any directory and will handle path configuration automatically.
"""

import sys
import os
from pathlib import Path

def setup_path():
    """Set up Python path to include the project directory"""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()

    # Add the project root to Python path
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    # Also add the color_pattern_approach directory
    color_approach_dir = script_dir / "color_pattern_approach"
    if color_approach_dir.exists() and str(color_approach_dir) not in sys.path:
        sys.path.insert(0, str(color_approach_dir))

def main():
    """Main function to launch color_pattern_visualizer.py"""
    setup_path()

    try:
        from color_pattern_approach.color_pattern_visualizer import main
        main()
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure you're running this from the project root directory")
        print("or that the project is installed as a package.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running color_pattern_visualizer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

