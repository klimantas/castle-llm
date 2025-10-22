import subprocess
import inspect
import time
import os
import json
from datetime import datetime
from pathlib import Path

from langchain_dartmouth.llms import ChatDartmouth
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from CybORG.Shared.Actions import *
from typing import Literal
from pydantic import BaseModel, Field
from pprint import pprint
from statistics import mean, stdev
from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers import BlueTableWrapper
from prettytable.prettytable import PrettyTable
from CybORG.Shared.Results import Results
from dotenv import load_dotenv
from tqdm import tqdm


def wrap(env):
    # return ChallengeWrapper(env=env, agent_name='Blue')
    return BlueTableWrapper(env=env, agent="Blue")


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


class Action(BaseModel):
    analyse: str = Field(description="Think step by step")
    action: Literal[
        "MONITOR",
        "ANALYSE",
        "DECOY_APACHE",
        "DECOY_FEMITTER",
        "DecoyHarakaSMPT",
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


SYSTEM_PROMPT = """
You are a Blue-team autonomous cyber defense agent in a simulated enterprise network under attack by a Red agent. Your goal is to prevent unauthorized access, preserve critical infrastructure, and minimize damage.

The network includes three subnets: Subnet 1 (non-critical), Subnet 2 (server), and Subnet 3 (operational). The Red agent starts on Subnet 1 and attempts to escalate access toward Subnet 3.

At each time-step (up to 50 total), you receive:
- Partial observations of the network state (e.g., host status, alerts, detected scans).
- The current score and time-step.
- Past action and state history.

You may perform one action per time-step from the following categories:

1. RECONNAISSANCE
   - MONITOR: Perform a broad scan to detect suspicious activity across the network.
   - ANALYSE(host): Perform a deep scan of a specific host to understand its status, open ports, or Red agent activity.

2. DECEPTION
   Deploy honeypots to trick or delay the Red agent. These actions depend on host configuration (OS, open ports):
   - DECOY_APACHE(host)
   - DECOY_FEMITTER(host)
   - DECOY_HARAKASMPT(host)
   - DECOY_SMSS(host)
   - DECOY_SSHD(host)
   - DECOY_SVCHOST(host)
   - DECOY_TOMCAT(host)
   - DECOY_VSFTPD(host)

3. RESTORATIVE
   - REMOVE(host): Attempt to remove Red presence from a host. Only succeeds if Red has NOT gained root access.
   - RESTORE(host): Fully resets a compromised host, even if Red has root access. This is more costly.

4. NO-OP
   - SLEEP: Do nothing this time-step.

Your goal is to:
- Prevent the Red agent from gaining root access, especially on critical infrastructure.
- Detect intrusions early using reconnaissance.
- Use decoys strategically to mislead Red.
- Remove Red presence when possible.
- Restore compromised hosts to maintain network integrity (Compromised value: Privileged).

When prompted for action, return exactly one line in this format:
ACTION_NAME(optional_host_argument)

Examples:
- MONITOR
- ANALYSE(Host_5)
- DECOY_SSHD(Host_2)
- REMOVE(Host_4)
- SLEEP

NOTE: The final reward is the accumulated score over all time-steps, which is the sum of the rewards received at each step.
Table 1 Blue rewards for red administrator access (per turn)
 
| Subnet   | Hosts              | Blue Reward for Red Access (per turn) |
|:--------:|:------------------:|:---------------------------------------:|
| Subnet 1 | User Hosts (User 0,1,2,3,4)| -0.1
| Subnet 2 | Enterprise Servers (Enterprise 0,1,2)  | -1
| Subnet 3 | Operational Server (Op_Server0) | -1
| Subnet 3 | Operational Hosts (Op_Host0, Op_Host1, Op_Host2) | -0.1

*Table 2 Blue rewards for successful red actions (per turn)*

| Agent    | Hosts              | Action   | Blue Reward (per turn) |
|:--------:|:------------------:|:--------:|:------------------------:|
| Red      | Operational Server (Op_Server0) | Impact   | -10
| Blue     | Any                | Restore  | -1


Tip:
When red agent gain roots access (Privileged value in the Compromised column), Blue agent should restore.
"""

HUMAN_PROMPT = """
History:
{history}

Action:
"""


# - Use REMOVE/RESTORE only when necessary, as they are costly.
def convert_model_output_to_action(output: Action, session=0):
    """
    Convert the model output string to an Action object.
    """
    if output.action == "SLEEP":
        return Sleep()
    elif output.action == "MONITOR":
        return Monitor(agent="Blue", session=session)
    elif output.action == "ANALYSE":
        return Analyse(hostname=output.host, agent="Blue", session=session)
    elif output.action == "DECOY_APACHE":
        return DecoyApache(hostname=output.host, agent="Blue", session=session)
    elif output.action == "DECOY_FEMITTER":
        return DecoyFemitter(hostname=output.host, agent="Blue", session=session)
    elif output.action == "DECOY_HARAKASMTP":
        return DecoyHarakaSMPT(hostname=output.host, agent="Blue", session=session)
    elif output.action == "DECOY_SMSS":
        return DecoySmss(hostname=output.host, agent="Blue", session=session)
    elif output.action == "DECOY_SSHD":
        return DecoySSHD(hostname=output.host, agent="Blue", session=session)
    elif output.action == "DECOY_SVCHOST":
        return DecoySvchost(hostname=output.host, agent="Blue", session=session)
    elif output.action == "DECOY_TOMCAT":
        return DecoyTomcat(hostname=output.host, agent="Blue", session=session)
    elif output.action == "DECOY_VSFTPD":
        return DecoyVsftpd(hostname=output.host, agent="Blue", session=session)
    elif output.action == "REMOVE":
        return Remove(hostname=output.host, agent="Blue", session=session)
    elif output.action == "RESTORE":
        return Restore(hostname=output.host, agent="Blue", session=session)
    return None


def run_episode(
    model="llama-3.3-70b",
    temperature=1.0,
    api_key=None,
    red_agent=RedMeanderAgent,
    steps=30,
):
    """Run a single episode and return the total reward, blue and red action list, and number of impacts."""
    if api_key is None:
        raise ValueError("API key must be provided")

    # Create ChatDartmouth LLM
    chat = ChatDartmouth(
        dartmouth_chat_api_key=api_key, model_name=model, temperature=temperature
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

    scenario = "Scenario2"
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"

    cyborg = CybORG(path, "sim", agents={"Red": red_agent})  # Comment out one
    env = BlueTableWrapper(cyborg)

    results = env.reset(agent="Blue")
    obs, reward, done, action_space = (
        results.observation,
        results.reward,
        results.done,
        results.action_space,
    )
    obs.del_column("Subnet")
    obs.del_column("IP Address")
    history = f"State: \n{str(obs)}"

    r = []
    blue_actions = []
    red_actions = []
    impacts = 0

    for i in tqdm(range(steps), desc="Episode steps", leave=False):
        # Use LangChain chain to get structured output
        try:
            output = chain.invoke(
                {
                    "history": history,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
        except Exception as e:
            print(f"\nError getting structured output: {e}")
            # Fallback to MONITOR action if parsing fails
            output = Action(
                analyse="Error occurred, monitoring network",
                action="MONITOR",
                host="nohost",
            )

        action = convert_model_output_to_action(output)

        results = env.step(action=action, agent="Blue")
        obs, reward, done, action_space = (
            results.observation,
            results.reward,
            results.done,
            results.action_space,
        )
        obs.del_column("Subnet")
        obs.del_column("IP Address")

        red_action = env.get_last_action("Red")
        red_actions.append(red_action)
        if "impact" in str(red_action).lower():
            impacts += 1

        history = f"{history}\nAction: {results.action}\nReward: {results.reward}\nState: {obs}"
        r.append(results.reward)
        blue_actions.append(results.action)

    return sum(r), r, blue_actions, red_actions, impacts


if __name__ == "__main__":
    # Load Dartmouth API key
    loaded = load_dotenv(dotenv_path="dartmouth_key.env")
    print("Loaded Dartmouth dotenv:", loaded)
    chat_key = os.getenv("DARTMOUTH_CHAT_API_KEY")
    print("Loaded Dartmouth Chat API Key:", chat_key)

    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Create experiment folder under "results" with current experiment version and timestamp
    experiment_version = "v0-0-0"
    yearMonth = datetime.now().strftime("%Y%m")

    experiment_dir = results_dir / f"results_{experiment_version}_{yearMonth}"
    experiment_dir.mkdir(exist_ok=True)

    # Dartmouth Chat models (adjust based on available models)
    models = [
        "openai.gpt-oss-120b",
        # Add other Dartmouth-supported models here
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

            total_reward, step_rewards, actions, red_actions, impacts = run_episode(
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
                }
            )

            print(f"\n✓ Episode {episode + 1} completed")
            print(f"  Total Reward: {total_reward:.2f}")
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
