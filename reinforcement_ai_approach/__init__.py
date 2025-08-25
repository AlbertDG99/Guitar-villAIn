"""
Reinforcement AI Approach - Guitar Hero AI using Deep Q-Learning
"""

__version__ = "1.0.0"
__author__ = "Guitar VillAIn Team"

# Export main modules for easier importing
from .input_preview import main as run_input_preview
from .train import main as run_training
from . import src

__all__ = [
    'run_input_preview',
    'run_training',
    'src'
]