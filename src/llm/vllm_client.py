"""vLLM client implementation for local model serving."""

import json
import requests
from typing import Dict, Any, Tuple, Optional
from .base import BaseLLMClient
from ..models import Action
from ..prompts import format_human_prompt


class VLLMClient(BaseLLMClient):
    """vLLM client for local model serving via HTTP API."""
    
    def __init__(self, model: str, temperature: float = 1.0, api_key: Optional[str] = None, base_url: str = "http://localhost:8000"):
        """Initialize the vLLM client.
        
        Args:
            model: Name of the model being served by vLLM.
            temperature: Sampling temperature.
            api_key: API key (optional for local serving).
            base_url: Base URL of the vLLM server.
        """
        super().__init__(model, temperature, api_key)
        self.base_url = base_url.rstrip('/')
        self.completions_url = f"{self.base_url}/v1/completions"
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        
    def get_action(
        self, 
        system_prompt: str, 
        history: str,
        step: int
    ) -> Tuple[Action, Optional[Dict[str, Any]]]:
        """Get an action from vLLM server.
        
        Args:
            system_prompt: The system prompt.
            history: The conversation history.
            step: Current step number (for logging).
            
        Returns:
            Tuple of (Action object, error dict if failed else None).
        """
        try:
            # Prepare messages for chat format
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": format_human_prompt(history)}
            ]
            
            # Prepare request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": 1024,
                "stop": None
            }
            
            # Add API key to headers if provided
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Make request to vLLM server
            response = requests.post(
                self.chat_url,
                json=payload,
                headers=headers,
                timeout=120  # 2-minute timeout
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            raw_content = response_data["choices"][0]["message"]["content"]
            
            print(f"\n[Step {step+1}] Raw vLLM Response:\n{raw_content}")
            
            # Process the response text
            processed_content = self._process_response(raw_content, step)
            
            # Parse JSON to Action object
            try:
                action_data = json.loads(processed_content)
                
                # Normalize action to uppercase
                if 'action' in action_data and isinstance(action_data['action'], str):
                    action_data['action'] = action_data['action'].upper()
                
                # Create Action object
                output = Action(**action_data)
                print(f"[Step {step+1}] Parsed: action={output.action}, host={output.host}")
                
                return output, None
                
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"[Step {step+1}] JSON Parse Error: {e}")
                return (
                    Action(action="MONITOR", host="nohost", reasoning="Fallback due to JSON parse error"),
                    {"step": step, "error": f"JSON parse error: {str(e)}", "error_type": "ParseError"}
                )
            
        except requests.exceptions.RequestException as e:
            print(f"\n[Step {step+1}] vLLM Request Error: {e}")
            return (
                Action(action="MONITOR", host="nohost", reasoning="Fallback due to network error"),
                {"step": step, "error": str(e), "error_type": "RequestError"}
            )
        except Exception as e:
            print(f"\n[Step {step+1}] vLLM Unexpected Error: {e}")
            return (
                Action(action="MONITOR", host="nohost", reasoning="Fallback due to unexpected error"),
                {"step": step, "error": str(e), "error_type": type(e).__name__}
            )
    
    def _process_response(self, text: str, step: int) -> str:
        """Process and clean the response text."""
        processed_text = text.strip()
        
        # Strip markdown code blocks if present
        if processed_text.startswith('```'):
            lines = processed_text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            processed_text = '\n'.join(lines)
            print(f"[Step {step+1}] Stripped markdown code blocks")
        
        # Extract JSON from response if it contains other text
        json_start = processed_text.find('{')
        json_end = processed_text.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_part = processed_text[json_start:json_end+1]
            # Validate that it's proper JSON
            try:
                json.loads(json_part)
                processed_text = json_part
                print(f"[Step {step+1}] Extracted JSON from response")
            except json.JSONDecodeError:
                pass  # Use original text if JSON extraction fails
        
        return processed_text
    
    def health_check(self) -> bool:
        """Check if the vLLM server is healthy and responsive."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except:
            return False