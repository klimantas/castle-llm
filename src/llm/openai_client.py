"""OpenAI LLM client implementation."""

from typing import Dict, Any, Tuple, Optional
from openai import OpenAI
from .base import BaseLLMClient
from ..models import Action
from ..prompts import format_human_prompt


class OpenAIClient(BaseLLMClient):
    """OpenAI LLM client using structured outputs."""
    
    def __init__(self, model: str, temperature: float = 1.0, api_key: Optional[str] = None):
        """Initialize the OpenAI client.
        
        Args:
            model: Name of the OpenAI model to use.
            temperature: Sampling temperature.
            api_key: OpenAI API key.
        """
        super().__init__(model, temperature, api_key)
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self.client = OpenAI(api_key=api_key)
    
    def get_action(
        self, 
        system_prompt: str, 
        history: str,
        step: int
    ) -> Tuple[Action, Optional[Dict[str, Any]]]:
        """Get an action from OpenAI.
        
        Args:
            system_prompt: The system prompt.
            history: The conversation history.
            step: Current step number (for logging).
            
        Returns:
            Tuple of (Action object, error dict if failed else None).
        """
        try:
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": format_human_prompt(history)}
            ]
            
            response = self.client.responses.parse(
                model=self.model,
                temperature=self.temperature,
                input=conversation,
                text_format=Action,
            )
            
            output = response.output_parsed
            
            if not output:
                return (
                    Action(action="MONITOR", host="nohost", reasoning="Fallback due to empty output"),
                    {"step": step, "error": "output_parsed is None or empty"}
                )
            
            print(f"\n[Step {step+1}] OpenAI Response: action={output.action}, host={output.host}")
            return output, None
            
        except Exception as e:
            print(f"\n[Step {step+1}] OpenAI Error: {e}")
            return (
                Action(action="MONITOR", host="nohost", reasoning="Fallback due to API error"),
                {"step": step, "error": str(e), "error_type": type(e).__name__}
            )
