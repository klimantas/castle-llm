"""Core agent logic for running episodes."""

from typing import Tuple, List, Dict, Any, Optional
from tqdm import tqdm

from .llm.base import BaseLLMClient
from .environment import initialize_environment, execute_step
from .actions import convert_action_to_cyborg, get_fallback_action
from .prompts import load_system_prompt
from .models import Action


class CyberDefenseAgent:
    """Blue team cyber defense agent using LLMs."""
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        scenario: str = "Scenario2",
        red_agent = None,
        print_observations: bool = False
    ):
        """Initialize the agent.
        
        Args:
            llm_client: The LLM client to use for decision making.
            scenario: Scenario name to load.
            red_agent: Red agent class to use as adversary.
            print_observations: Whether to print observation tables.
        """
        self.llm_client = llm_client
        self.scenario = scenario
        self.red_agent = red_agent
        self.print_observations = print_observations
        self.system_prompt = load_system_prompt()
    
    def run_episode(
        self,
        steps: int = 30
    ) -> Tuple[float, List[float], List[Any], List[Any], int, List[Dict], List[Dict]]:
        """Run a single episode.
        
        Args:
            steps: Number of steps to run.
            
        Returns:
            Tuple containing:
                - total_reward: Sum of all step rewards
                - step_rewards: List of rewards per step
                - blue_actions: List of blue agent actions
                - red_actions: List of red agent actions
                - impacts: Number of impact actions by red agent
                - api_failures: List of API failure details
                - parsing_failures: List of parsing failure details
        """
        # Initialize environment
        env, history = initialize_environment(
            scenario=self.scenario,
            red_agent=self.red_agent
        )
        
        # Track results
        step_rewards = []
        blue_actions = []
        red_actions = []
        impacts = 0
        api_failures = []
        parsing_failures = []
        
        # Run episode
        for i in tqdm(range(steps), desc="Episode steps", leave=False):
            # Get action from LLM
            output, error = self.llm_client.get_action(
                system_prompt=self.system_prompt,
                history=history,
                step=i
            )
            
            if error:
                api_failures.append(error)
            
            # Convert to CybORG action
            action = convert_action_to_cyborg(output)
            
            if action is None:
                parsing_failures.append({
                    "step": i,
                    "output_action": output.action,
                    "output_host": output.host,
                    "output_reasoning": output.reasoning,
                })
                print(f"\n[Step {i+1}] Action conversion failed, using fallback")
                action = get_fallback_action()
            
            # Execute step
            history, reward, blue_action, red_action = execute_step(
                env=env,
                action=action,
                history=history,
                print_table=self.print_observations
            )
            
            # Track results
            step_rewards.append(reward)
            blue_actions.append(blue_action)
            red_actions.append(red_action)
            
            if "impact" in str(red_action).lower():
                impacts += 1
        
        total_reward = sum(step_rewards)
        
        return (
            total_reward,
            step_rewards,
            blue_actions,
            red_actions,
            impacts,
            api_failures,
            parsing_failures
        )
