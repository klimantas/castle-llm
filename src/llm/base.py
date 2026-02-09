"""Base abstract class for LLM clients."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
from ..models import Action


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model: str, temperature: float = 1.0, api_key: Optional[str] = None):
        """Initialize the LLM client.
        
        Args:
            model: Name of the model to use.
            temperature: Sampling temperature.
            api_key: API key for authentication.
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        
    @abstractmethod
    def get_action(
        self, 
        system_prompt: str, 
        history: str,
        step: int
    ) -> Tuple[Action, Optional[Dict[str, Any]]]:
        """Get an action from the LLM.
        
        Args:
            system_prompt: The system prompt.
            history: The conversation history.
            step: Current step number (for logging).
            
        Returns:
            Tuple of (Action object, error dict if failed else None).
        """
        pass
