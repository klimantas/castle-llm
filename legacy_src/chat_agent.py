import inspect
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Tuple, List, Dict, Any

from langchain_dartmouth.llms import ChatDartmouth
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from CybORG.Shared.Actions import *
from pydantic import BaseModel, Field, constr
from statistics import mean, stdev
from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers import BlueTableWrapper
from dotenv import load_dotenv
from tqdm import tqdm


# Configuration constants
DEFAULT_AGENT = "Blue"
DEFAULT_SESSION = 0
DEFAULT_STEPS = 30
DEFAULT_EPISODES = 3
DEFAULT_TEMPERATURE = 1.0
SCENARIO_FILE = "Scenario2"
MAX_HISTORY_LENGTH = 8000  # Prevent prompt from growing too large


def wrap(env: Any) -> BlueTableWrapper:
    """Wrap a CybORG environment with BlueTableWrapper.
    
    Args:
        env: The CybORG environment to wrap.
        
    Returns:
        A BlueTableWrapper instance wrapping the environment.
    """
    return BlueTableWrapper(env=env, agent="Blue")


def load_system_prompt(prompt_file: str = "prompts/system_prompt.txt") -> str:
    """Load the system prompt from a file.
    
    Args:
        prompt_file: Path to the system prompt file relative to script location.
        
    Returns:
        The system prompt as a string.
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist.
    """
    prompt_path = Path(__file__).parent / prompt_file
    with open(prompt_path, 'r') as f:
        return f.read()


class Action(BaseModel):
    reasoning: Optional[constr(max_length=200)] = Field(
        None, description="Optional short reasoning (<=200 chars)"
    )
    action: Literal[
        "MONITOR",
        "ANALYSE",
        "DECOY_APACHE",
        "DECOY_FEMITTER",
        "DECOY_HARAKASMPT",
        "DECOY_SMSS",
        "DECOY_SSHD",
        "DECOY_SVCHOST",
        "DECOY_TOMCAT",
        "DECOY_VSFTPD",
        "REMOVE",
        "RESTORE",
        "SLEEP",
    ]
    host: Literal[
        "Defender",
        "Enterprise0",
        "Enterprise1",
        "Enterprise2",
        "Op_Host0",
        "Op_Host1",
        "Op_Host2",
        "Op_Server0",
        "User0",
        "User1",
        "User2",
        "User3",
        "User4",
        "nohost",
    ] = Field(
        default="nohost",
        description="Host to perform the action on. Default is 'nohost' for actions that do not require a host argument.",
    )


# Load prompts from files
SYSTEM_PROMPT = load_system_prompt()

HUMAN_PROMPT = """
History:
{history}

Action:
"""


ACTION_MAP = {
    "SLEEP": lambda h, s: Sleep(),
    "MONITOR": lambda h, s: Monitor(agent="Blue", session=s),
    "ANALYSE": lambda h, s: Analyse(hostname=h, agent="Blue", session=s),
    "DECOY_APACHE": lambda h, s: DecoyApache(hostname=h, agent="Blue", session=s),
    "DECOY_FEMITTER": lambda h, s: DecoyFemitter(hostname=h, agent="Blue", session=s),
    "DECOY_HARAKASMTP": lambda h, s: DecoyHarakaSMPT(hostname=h, agent="Blue", session=s),
    "DECOY_SMSS": lambda h, s: DecoySmss(hostname=h, agent="Blue", session=s),
    "DECOY_SSHD": lambda h, s: DecoySSHD(hostname=h, agent="Blue", session=s),
    "DECOY_SVCHOST": lambda h, s: DecoySvchost(hostname=h, agent="Blue", session=s),
    "DECOY_TOMCAT": lambda h, s: DecoyTomcat(hostname=h, agent="Blue", session=s),
    "DECOY_VSFTPD": lambda h, s: DecoyVsftpd(hostname=h, agent="Blue", session=s),
    "REMOVE": lambda h, s: Remove(hostname=h, agent="Blue", session=s),
    "RESTORE": lambda h, s: Restore(hostname=h, agent="Blue", session=s),
}


def convert_model_output_to_action(output: Action, session: int = 0) -> Optional[Any]:
    """Convert a parsed Action model to a CybORG action.
    
    Args:
        output: Pydantic Action model containing action type and host.
        session: Session ID for the action (default: 0).
        
    Returns:
        A CybORG action object, or None if the action type is invalid.
    """
    action_factory = ACTION_MAP.get(output.action.upper())
    return action_factory(output.host, session) if action_factory else None


def initialize_environment(scenario: str = "Scenario2", red_agent=RedMeanderAgent) -> Tuple[BlueTableWrapper, str]:
    """Initialize the CybORG environment and return the wrapped environment and initial observation.
    
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


def parse_llm_response(
    response_text: str, 
    parser: PydanticOutputParser, 
    step: int
) -> Tuple[Action, Optional[Dict[str, Any]]]:
    """Parse and normalize LLM response text into an Action object.
    
    Args:
        response_text: Raw text response from the LLM.
        parser: Pydantic parser for Action objects.
        step: Current step number (for logging).
        
    Returns:
        Tuple of (parsed Action object, error dict if parsing failed else None).
    """
    print(f"\n[Step {step+1}] Raw LLM Response:\n{response_text}")
    
    # Strip markdown code blocks if present
    if response_text.strip().startswith('```'):
        lines = response_text.strip().split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        response_text = '\n'.join(lines)
        print(f"[Step {step+1}] Stripped markdown, cleaned text:\n{response_text}")
    
    # Normalize the action field to uppercase before parsing
    try:
        response_json = json.loads(response_text)
        print(f"[Step {step+1}] Parsed JSON before normalization: {response_json}")
        if 'action' in response_json and isinstance(response_json['action'], str):
            response_json['action'] = response_json['action'].upper()
            print(f"[Step {step+1}] Normalized action to: {response_json['action']}")
        response_text = json.dumps(response_json)
    except json.JSONDecodeError as e:
        print(f"[Step {step+1}] JSONDecodeError: {e}")
    
    # Parse the response
    output = parser.parse(response_text)
    print(f"[Step {step+1}] Parsed: action={output.action}, host={output.host}")
    
    return output, None


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


def run_episode(
    model: str = "openai.gpt-oss-120b",
    temperature: float = 1.0,
    api_key: Optional[str] = None,
    red_agent = RedMeanderAgent,
    steps: int = 30,
    print_observations: bool = False,
) -> Tuple[float, List[float], List[Any], List[Any], int, List[Dict], List[Dict]]:
    """Run a single episode of the blue agent defending against a red agent.
    
    Args:
        model: Name of the LLM model to use.
        temperature: Sampling temperature for the LLM.
        api_key: API key for the Dartmouth Chat service.
        red_agent: Red agent class to use as adversary.
        steps: Number of steps to run in the episode.
        print_observations: Whether to print formatted observation tables to console.
        
    Returns:
        Tuple containing:
            - total_reward: Sum of all step rewards
            - step_rewards: List of rewards per step
            - blue_actions: List of blue agent actions
            - red_actions: List of red agent actions
            - impacts: Number of impact actions by red agent
            - api_failures: List of API failure details
            - parsing_failures: List of parsing failure details
            
    Raises:
        ValueError: If api_key is None.
    """
    if api_key is None:
        raise ValueError("API key must be provided")

    # Create ChatDartmouth LLM
    chat = ChatDartmouth(
        dartmouth_chat_api_key=api_key,
        model_name=model,
        temperature=temperature,
        max_tokens=8192,
    )

    # Create output parser for structured output
    parser = PydanticOutputParser(pydantic_object=Action)

    # Create prompt template with format instructions
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT + "\n\n{format_instructions}"),
            ("user", HUMAN_PROMPT),
        ]
    )

    # Create the chain: prompt -> llm -> parser
    chain = prompt | chat | parser

    # Initialize environment
    env, history = initialize_environment(scenario="Scenario2", red_agent=red_agent)

    r = []
    blue_actions = []
    red_actions = []
    impacts = 0
    
    # Track parsing failures
    api_failures = []
    parsing_failures = []

    for i in tqdm(range(steps), desc="Episode steps", leave=False):
        # Get LLM response
        try:
            raw_response = chat.invoke(
                prompt.format_messages(
                    history=history,
                    format_instructions=parser.get_format_instructions()
                )
            )
            response_text = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
            
            # Parse and normalize response
            output, parse_error = parse_llm_response(response_text, parser, i)
            
        except Exception as e:
            print(f"\n[Step {i+1}] Error getting structured output: {e}")
            api_failures.append({
                "step": i,
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Fallback to MONITOR action if parsing fails
            output = Action(
                reasoning=None,
                action="MONITOR",
                host="nohost",
            )

        # Convert to CybORG action
        action = convert_model_output_to_action(output)
        
        # Check if action conversion failed
        if action is None:
            parsing_failures.append({
                "step": i,
                "output_action": output.action if output else "None",
                "output_host": output.host if output else "None",
                "output_reasoning": output.reasoning if output else "None",
            })
            print(f"\n[Step {i+1}] Action conversion failed, using MONITOR as fallback")
            action = Monitor(agent="Blue", session=0)

        # Execute step in environment
        history, reward, blue_action, red_action = execute_step(env, action, history, print_table=print_observations)
        
        # Track results
        r.append(reward)
        blue_actions.append(blue_action)
        red_actions.append(red_action)
        if "impact" in str(red_action).lower():
            impacts += 1

    return sum(r), r, blue_actions, red_actions, impacts, api_failures, parsing_failures


if __name__ == "__main__":
    # Load Dartmouth API key
    loaded = load_dotenv(dotenv_path="dartmouth_key.env")
    print("Loaded Dartmouth dotenv:", loaded)
    chat_key = os.getenv("DARTMOUTH_CHAT_API_KEY")
    print(f"Loaded Dartmouth Chat API Key: {chat_key}")

    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Create experiment folder under "results" with current experiment version and timestamp
    experiment_version = "v0-0-1"
    yearMonth = datetime.now().strftime("%Y%m")

    experiment_dir = results_dir / f"results_{experiment_version}_{yearMonth}"
    experiment_dir.mkdir(exist_ok=True)

    # Dartmouth Chat models (adjust based on available models, free models only)
    models = [
        "openai.gpt-oss-120b",
        "google.gemma-3-27b-it",
    ]

    for model in models:
        temp = 1.0
        red_agent = RedMeanderAgent
        # red_agent = B_lineAgent

        episodes = 3
        steps = 30
        all_rewards = []
        all_impacts = []

        # Prepare results log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_log = {
            "timestamp": timestamp,
            "scenario": "Scenario2",
            "episodes": episodes,
            "steps_per_episode": steps,
            "model": model,
            "temperature": temp,
            "red_agent": (
                "RedMeanderAgent" if red_agent == RedMeanderAgent else "B_lineAgent"
            ),
            "episodes": [],
        }

        print(f"\n{'='*60}")
        print(f"Running {episodes} episodes of {model}")
        print(f"{'='*60}\n")

        for episode in range(episodes):
            print(f"\n{'─'*60}")
            print(f"Episode {episode + 1}/{episodes}")
            print(f"{'─'*60}")

            total_reward, step_rewards, actions, red_actions, impacts, api_failures, parsing_failures = run_episode(
                model=model,
                temperature=temp,
                api_key=chat_key,  # Use Dartmouth Chat API key instead of OpenAI key
                red_agent=red_agent,
                steps=steps,
            )
            all_rewards.append(total_reward)
            all_impacts.append(impacts)

            # Store run data
            results_log["episodes"].append(
                {
                    "episode_number": episode + 1,
                    "total_reward": total_reward,
                    "step_rewards": step_rewards,
                    "actions": [str(a) for a in actions],
                    "red actions": [str(a) for a in red_actions],
                    "impacts": impacts,
                    "api_failures": api_failures,
                    "parsing_failures": parsing_failures,
                    "api_failure_count": len(api_failures),
                    "parsing_failure_count": len(parsing_failures),
                    "failure_rate": (len(api_failures) + len(parsing_failures)) / steps,
                }
            )

            print(f"\n✓ Episode {episode + 1} completed")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  API Failures: {len(api_failures)}/{steps}")
            print(f"  Parsing Failures: {len(parsing_failures)}/{steps}")
            print(f"  Step Rewards: {step_rewards}")

        # Calculate statistics
        results_log["statistics"] = {
            "mean_reward": mean(all_rewards),
            "stdev_reward": stdev(all_rewards) if len(all_rewards) > 1 else None,
            "min_reward": min(all_rewards),
            "max_reward": max(all_rewards),
            "all_total_rewards": all_rewards,
            "mean_impacts": mean(all_impacts),
        }

        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"All Total Rewards: {all_rewards}")
        print(f"Average Reward: {results_log['statistics']['mean_reward']:.2f}")
        if results_log["statistics"]["stdev_reward"] is not None:
            print(f"Std Dev: {results_log['statistics']['stdev_reward']:.2f}")
        print(f"Min Reward: {results_log['statistics']['min_reward']:.2f}")
        print(f"Max Reward: {results_log['statistics']['max_reward']:.2f}")
        print(f"Average Impacts: {results_log['statistics']['mean_impacts']:.2f}")

        # Save to file
        filename = experiment_dir / f"{timestamp}_LLM_Agent.json"
        with open(filename, "w") as f:
            json.dump(results_log, f, indent=2)

        print(f"\n✓ Results saved to: {filename}")
        print(f"{'='*60}\n")
