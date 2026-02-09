"""Prompt loading and management."""

from pathlib import Path
from typing import Optional
from .config import PROMPTS_DIR


HUMAN_PROMPT_TEMPLATE = """
History:
{history}

Action:
"""


def load_system_prompt(prompt_file: str = "system_prompt.txt") -> str:
    """Load the system prompt from a file.
    
    Args:
        prompt_file: Name of the prompt file in the prompts directory.
        
    Returns:
        The system prompt as a string.
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist.
    """
    prompt_path = PROMPTS_DIR / prompt_file
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r') as f:
        return f.read()


def format_human_prompt(history: str) -> str:
    """Format the human prompt with the current history.
    
    Args:
        history: The conversation history to include.
        
    Returns:
        Formatted human prompt.
    """
    return HUMAN_PROMPT_TEMPLATE.format(history=history)


def truncate_history(history: str, max_length: int = 8000) -> str:
    """Truncate history if it exceeds max length.
    
    Args:
        history: The history string to truncate.
        max_length: Maximum allowed length.
        
    Returns:
        Truncated history string.
    """
    if len(history) <= max_length:
        return history
    
    # Keep most recent entries
    return "...[earlier history truncated]\n" + history[-max_length:]
