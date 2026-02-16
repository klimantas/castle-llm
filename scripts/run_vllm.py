#!/usr/bin/env python3
"""Run evaluation using vLLM local model serving."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import VLLMClient
from src.agent import CyberDefenseAgent
from src.evaluation import EvaluationRunner
from CybORG.Agents import B_lineAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent


def main():
    """Main entry point for vLLM evaluation."""
    # Load API key (optional for local vLLM serving)
    load_dotenv(dotenv_path="vllm_key.env")
    api_key = os.getenv("VLLM_API_KEY")  # Optional
    
    # vLLM server configuration
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
    
    print(f"Using vLLM server at: {base_url}")
    if api_key:
        print(f"Using API Key: {api_key[:10]}...")
    else:
        print("No API key provided (using local serving)")
    
    # Configuration - popular open-source models for vLLM
    models = [
        "meta-llama/Llama-3.2-3B-Instruct",
        "microsoft/Phi-3.5-mini-instruct",
        # "mistralai/Mistral-7B-Instruct-v0.3",
        # "teknium/OpenHermes-2.5-Mistral-7B",
    ]
    
    temperature = 1.0
    red_agent = RedMeanderAgent
    # red_agent = B_lineAgent
    red_agent_name = red_agent.__name__
    
    episodes = 3
    steps = 30
    
    # Check server health first
    print("\nChecking vLLM server health...")
    test_client = VLLMClient(
        model="test",
        temperature=temperature,
        api_key=api_key,
        base_url=base_url
    )
    
    if not test_client.health_check():
        print("ERROR: vLLM server is not reachable!")
        print(f"Make sure vLLM is running at {base_url}")
        print("Example command to start vLLM server:")
        print("vllm serve meta-llama/Llama-3.2-3B-Instruct --host 0.0.0.0 --port 8000")
        return
    
    print("✓ vLLM server is healthy")
    
    # Run evaluation for each model
    for model in models:
        print(f"\n{'='*60}")
        print(f"Evaluating {model}")
        print(f"{'='*60}")
        
        # Create LLM client
        llm_client = VLLMClient(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url
        )
        
        # Create agent
        agent = CyberDefenseAgent(
            llm_client=llm_client,
            scenario="Scenario2",
            red_agent=red_agent,
            print_observations=False
        )
        
        # Run evaluation
        runner = EvaluationRunner(agent, experiment_version="v0-0-2")
        runner.run_evaluation(
            model_name=model.split('/')[-1],  # Use just the model name part
            temperature=temperature,
            red_agent_name=red_agent_name,
            episodes=episodes,
            steps_per_episode=steps
        )


if __name__ == "__main__":
    main()