"""
Color Pattern Approach - Guitar Hero AI using color detection
"""

__version__ = "1.0.0"
__author__ = "Guitar VillAIn Team"

# Export main classes for easier importing
from .color_pattern_visualizer import main as run_visualizer
from .config_manager import ConfigManager
from .screen_capture import ScreenCapture

__all__ = [
    'run_visualizer',
    'ConfigManager',
    'ScreenCapture'
]