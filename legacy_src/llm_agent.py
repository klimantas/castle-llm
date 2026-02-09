import subprocess
import inspect
import time
import os
import json
from datetime import datetime
from pathlib import Path

from openai import OpenAI

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

# Using Pydantic to define the expected structure of the model output
class Action(BaseModel):
    reasoning: str = Field(description="Explain your decision-making process step by step")
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

IMPORTANT: You must respond with a valid JSON object matching the provided schema. The response will be automatically parsed into the required format.

Example response:
{{"action": "MONITOR", "host": "nohost", "reasoning": "Scanning network for threats"}}

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
    action_upper = output.action.upper()
    
    if action_upper == "SLEEP":
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
    model="gpt-4.1",
    temperature=1.0,
    api_key=None,
    red_agent=RedMeanderAgent,
    steps=30,
):
    """Run a single episode and return the total reward, blue and red action list, and number of impacts."""
    if api_key is None:
        raise ValueError("API key must be provided")
    client = OpenAI(api_key=api_key)

    scenario = "Scenario2"
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"

    cyborg = CybORG(path, "sim", agents={"Red": red_agent})  
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
    print(f"\nInitial Observation:")
    print(obs)
    history = f"State: \n{str(obs)}"

    r = []
    blue_actions = []
    red_actions = []
    impacts = 0

    # Track parsing failures
    api_failures = []
    parsing_failures = []

    for i in tqdm(range(steps), desc="Episode steps", leave=False):
        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        conversation.extend(
            [{"role": "user", "content": HUMAN_PROMPT.format(history=history)}]
        )

        try:
            response = client.responses.parse(
                model=model,
                temperature=1.0,
                input=conversation,
                text_format=Action,
            )

            output = response.output_parsed
            print(f"\nPlaying against Red Agent: {red_agent.__name__}")
            # Log raw output for debugging
            if not output:
                api_failures.append({
                    "step": i,
                    "error": "output_parsed is None or empty",
                    "response": str(response),
                })
                # Fallback to safe action
                output = Action(action="MONITOR", host="nohost", reasoning="Fallback due to empty output")
            else:
                print(f"Step {i} model output:")
                pprint(output.model_dump())
                
            action = convert_model_output_to_action(output)

            # Check if action conversion was successful
            if action is None:
                parsing_failures.append({
                    "step": i,
                    "output_action": output.action if output else "None",
                    "output_host": output.host if output else "None",
                    "output_reasoning": output.reasoning if output else "None",
                })
                print(f"Parsing failure at step {i}: {output.model_dump() if output else 'No output'}")
                # Fallback to safe action
                action = Monitor(agent="Blue", session=0)
                
        except Exception as e:
            api_failures.append({
                "step": i,
                "error": str(e),
                "error_type": type(e).__name__
            })
            print(f"\nError at step {i}: {e}")
            # Fallback to safe action
            action = Monitor(agent="Blue", session=0)
            output = None

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

        r.append(results.reward)
        blue_actions.append(results.action)
        
        # Count compromised hosts - PrettyTable object, need to parse rows
        compromised_count = 0
        try:
            # PrettyTable stores data in _rows attribute
            for row in obs._rows:
                # Compromised column is typically at index 1 (after Hostname)
                compromised_value = row[1] if len(row) > 1 else None
                if compromised_value and compromised_value not in ['None', '', None]:
                    compromised_count += 1
        except Exception as e:
            # Fallback to string counting if table structure is different
            obs_str = str(obs)
            print(f"Error parsing table rows: {e}. Falling back to string parsing.")
            # Count only in Compromised column to avoid counting hostnames
            if 'Compromised' in obs_str:
                compromised_count = obs_str.count('| User ') + obs_str.count('| Privileged ')
        
        print(f"\nObservation after step {i}:")
        print(f"  Blue Action: {results.action}")
        print(f"  Red Action: {red_action}")
        print(f"  Reward: {results.reward}")
        print(f"  Cumulative Reward: {sum(r):.2f}")
        print(f"  Compromised Hosts: {compromised_count}")
        print(f"  History Length: {len(history)} chars")
        print(f"  State:\n{obs}")

        history = f"{history}\nAction: {results.action}\nReward: {results.reward}\nState: {obs}"

    return sum(r), r, blue_actions, red_actions, impacts, api_failures, parsing_failures


if __name__ == "__main__":
    loaded = load_dotenv(dotenv_path="openai_key.env")
    print("Loaded OpenAI dotenv:", loaded)
    api_key = os.getenv("OPENAI_API_KEY")
    print("Loaded API Key:", api_key)

    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Create experiment folder under "results" with current experiment version and timestamp
    experiment_version = "v0-0-2"
    yearMonth = datetime.now().strftime("%Y%m")

    experiment_dir = results_dir / f"results_{experiment_version}_{yearMonth}"
    experiment_dir.mkdir(exist_ok=True)

    models = [
        "gpt-5-nano",
        # "gpt-4.1-nano",
        # "gpt-4o-mini",
        # "gpt-5-mini",
        # "gpt-4.1-mini",
        # "gpt-4.1",
    ]
    for model in models:
        temp = 1.0
        # red_agent = RedMeanderAgent
        red_agent = B_lineAgent

        episodes = 1
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
                api_key=api_key,
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
