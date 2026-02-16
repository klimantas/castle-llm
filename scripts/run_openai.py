#!/usr/bin/env python3
"""Run evaluation using OpenAI models."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import OpenAIClient
from src.agent import CyberDefenseAgent
from src.evaluation import EvaluationRunner
from CybORG.Agents import B_lineAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent


def main():
    """Main entry point for OpenAI evaluation."""
    # Load API key
    load_dotenv(dotenv_path="openai_key.env")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        return
    
    print(f"Loaded OpenAI API Key: {api_key[:10]}...")
    
    # Configuration
    models = [
        "gpt-5-nano",
        # "gpt-4.1",
        # "gpt-4o-mini",
    ]
    
    temperature = 1.0
    red_agent = B_lineAgent
    # red_agent = RedMeanderAgent
    red_agent_name = red_agent.__name__
    
    episodes = 1
    steps = 30
    
    # Run evaluation for each model
    for model in models:
        print(f"\n{'='*60}")
        print(f"Evaluating {model}")
        print(f"{'='*60}")
        
        # Create LLM client
        llm_client = OpenAIClient(
            model=model,
            temperature=temperature,
            api_key=api_key
        )
        
        # Create agent
        agent = CyberDefenseAgent(
            llm_client=llm_client,
            scenario="Scenario2",
            red_agent=red_agent,
            print_observations=False
        )
        
        # Run evaluation
        runner = EvaluationRunner(agent, experiment_version="v0-0-3")
        runner.run_evaluation(
            model_name=model,
            temperature=temperature,
            red_agent_name=red_agent_name,
            episodes=episodes,
            steps_per_episode=steps
        )


if __name__ == "__main__":
    main()
