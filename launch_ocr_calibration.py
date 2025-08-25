#!/usr/bin/env python3
"""
Launcher script for OCR Calibration Tool
"""

import sys
from pathlib import Path

# Add the reinforcement_ai_approach directory to Python path
project_root = Path(__file__).parent
ai_approach_path = project_root / "reinforcement_ai_approach"
sys.path.insert(0, str(ai_approach_path))
sys.path.insert(0, str(ai_approach_path / "src"))

# Import and run the OCR calibrator
from ocr_calibration import main

if __name__ == "__main__":
    main()


