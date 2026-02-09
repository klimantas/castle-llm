"""CybORG environment initialization and management."""

import inspect
from typing import Tuple, Any
from CybORG import CybORG
from CybORG.Agents.Wrappers import BlueTableWrapper
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent


def initialize_environment(
    scenario: str = "Scenario2",
    red_agent = RedMeanderAgent
) -> Tuple[BlueTableWrapper, str]:
    """Initialize the CybORG environment.
    
    Args:
        scenario: Name of the scenario YAML file to load.
        red_agent: Red agent class to use as the adversary.
        
    Returns:
        Tuple of (wrapped environment, initial history string).
    """
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"

    cyborg = CybORG(path, "sim", agents={"Red": red_agent})
    env = BlueTableWrapper(cyborg)

    results = env.reset(agent="Blue")
    obs = results.observation
    obs.del_column("Subnet")
    obs.del_column("IP Address")
    history = f"State: \n{str(obs)}"
    
    return env, history


def execute_step(
    env: BlueTableWrapper,
    action: Any,
    history: str,
    print_table: bool = False
) -> Tuple[str, float, Any, Any]:
    """Execute one step in the environment.
    
    Args:
        env: The wrapped CybORG environment.
        action: The action to execute.
        history: Current history string.
        print_table: Whether to print the observation table to console.
        
    Returns:
        Tuple of (updated history, reward, blue action, red action).
    """
    results = env.step(action=action, agent="Blue")
    obs = results.observation
    obs.del_column("Subnet")
    obs.del_column("IP Address")
    
    red_action = env.get_last_action("Red")
    
    # Optionally print formatted table for better visualization
    if print_table:
        print(f"\n{'='*70}")
        print(f"Observation Table:")
        print(f"{'='*70}")
        print(obs)
        print(f"{'='*70}\n")
    
    new_history = f"{history}\nAction: {results.action}\nReward: {results.reward}\nState: {obs}"
    
    return new_history, results.reward, results.action, red_action
