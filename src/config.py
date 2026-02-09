"""Configuration constants for the Castle LLM project."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"

# Agent configuration
DEFAULT_AGENT = "Blue"
DEFAULT_SESSION = 0
DEFAULT_STEPS = 30
DEFAULT_EPISODES = 30
DEFAULT_TEMPERATURE = 1.0
SCENARIO_FILE = "Scenario2"

# Prompt configuration
MAX_HISTORY_LENGTH = 8000  # Prevent prompt from growing too large
