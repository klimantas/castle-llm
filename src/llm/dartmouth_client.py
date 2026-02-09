"""Dartmouth Chat LLM client implementation using LangChain."""

import json
from typing import Dict, Any, Tuple, Optional
from langchain_dartmouth.llms import ChatDartmouth
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from .base import BaseLLMClient
from ..models import Action
from ..prompts import format_human_prompt


class DartmouthClient(BaseLLMClient):
    """Dartmouth Chat LLM client using LangChain."""
    
    def __init__(self, model: str, temperature: float = 1.0, api_key: Optional[str] = None):
        """Initialize the Dartmouth client.
        
        Args:
            model: Name of the Dartmouth model to use.
            temperature: Sampling temperature.
            api_key: Dartmouth Chat API key.
        """
        super().__init__(model, temperature, api_key)
        if not api_key:
            raise ValueError("Dartmouth Chat API key is required")
        
        self.chat = ChatDartmouth(
            dartmouth_chat_api_key=api_key,
            model_name=model,
            temperature=temperature,
            max_tokens=8192,
        )
        
        self.parser = PydanticOutputParser(pydantic_object=Action)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}\n\n{format_instructions}"),
            ("user", "{human_prompt}"),
        ])
        self.chain = self.prompt | self.chat | self.parser
    
    def get_action(
        self, 
        system_prompt: str, 
        history: str,
        step: int
    ) -> Tuple[Action, Optional[Dict[str, Any]]]:
        """Get an action from Dartmouth Chat.
        
        Args:
            system_prompt: The system prompt.
            history: The conversation history.
            step: Current step number (for logging).
            
        Returns:
            Tuple of (Action object, error dict if failed else None).
        """
        try:
            raw_response = self.chat.invoke(
                self.prompt.format_messages(
                    system_prompt=system_prompt,
                    format_instructions=self.parser.get_format_instructions(),
                    human_prompt=format_human_prompt(history)
                )
            )
            response_text = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
            
            print(f"\n[Step {step+1}] Raw LLM Response:\n{response_text}")
            
            # Strip markdown code blocks if present
            response_text = self._strip_markdown(response_text, step)
            
            # Normalize action to uppercase
            response_text = self._normalize_action(response_text, step)
            
            # Parse the response
            output = self.parser.parse(response_text)
            print(f"[Step {step+1}] Parsed: action={output.action}, host={output.host}")
            
            return output, None
            
        except Exception as e:
            print(f"\n[Step {step+1}] Dartmouth Error: {e}")
            return (
                Action(action="MONITOR", host="nohost", reasoning="Fallback due to API error"),
                {"step": step, "error": str(e), "error_type": type(e).__name__}
            )
    
    def _strip_markdown(self, text: str, step: int) -> str:
        """Strip markdown code blocks from response."""
        if text.strip().startswith('```'):
            lines = text.strip().split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            text = '\n'.join(lines)
            print(f"[Step {step+1}] Stripped markdown")
        return text
    
    def _normalize_action(self, text: str, step: int) -> str:
        """Normalize action field to uppercase."""
        try:
            response_json = json.loads(text)
            print(f"[Step {step+1}] Parsed JSON: {response_json}")
            if 'action' in response_json and isinstance(response_json['action'], str):
                response_json['action'] = response_json['action'].upper()
                print(f"[Step {step+1}] Normalized action to: {response_json['action']}")
            return json.dumps(response_json)
        except json.JSONDecodeError as e:
            print(f"[Step {step+1}] JSONDecodeError: {e}")
            return text
