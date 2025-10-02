import subprocess
import inspect
import time
import os

from CybORG.Shared.Actions import *
from openai import OpenAI
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

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
def wrap(env):
    # return ChallengeWrapper(env=env, agent_name='Blue')
    return BlueTableWrapper(env=env, agent='Blue')

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

class Action(BaseModel):
    analyse: str = Field(description="Think step by step")
    action: Literal["MONITOR", "ANALYSE", "DECOY_APACHE", "DECOY_FEMITTER", "DecoyHarakaSMPT", "DECOY_SMSS", "DECOY_SSHD", "DECOY_SVCHOST", "DECOY_TOMCAT", "DECOY_VSFTPD", "REMOVE", "RESTORE", "SLEEP"]
    host: Literal["Enterprise0", "Enterprise1", "Enterprise2", "Op_Host0", "Op_Host1", "Op_Host2", "Op_Server0", "User0", "User1", "User2", "User3", "User4", "nohost"] = Field(default="nohost", description="Host to perform the action on. Default is 'nohost' for actions that do not require a host argument.")

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
        return Monitor(agent='Blue', session=session)   
    elif output.action == "ANALYSE":
        return Analyse(hostname=output.host, agent='Blue', session=session)
    elif output.action == "DECOY_APACHE":
        return DecoyApache(hostname=output.host, agent='Blue', session=session)
    elif output.action == "DECOY_FEMITTER":   
        return DecoyFemitter(hostname=output.host, agent='Blue', session=session)
    elif output.action == "DECOY_HARAKASMTP":
        return DecoyHarakaSMPT(hostname=output.host, agent='Blue', session=session)
    elif output.action == "DECOY_SMSS":
        return DecoySmss(hostname=output.host, agent='Blue', session=session)
    elif output.action == "DECOY_SSHD":
        return DecoySSHD(hostname=output.host, agent='Blue', session=session)
    elif output.action == "DECOY_SVCHOST":
        return DecoySvchost(hostname=output.host, agent='Blue', session=session)
    elif output.action == "DECOY_TOMCAT":
        return DecoyTomcat(hostname=output.host, agent='Blue', session=session)
    elif output.action == "DECOY_VSFTPD":
        return DecoyVsftpd(hostname=output.host, agent='Blue', session=session)
    elif output.action == "REMOVE":
        return Remove(hostname=output.host, agent='Blue', session=session)
    elif output.action == "RESTORE":
        return Restore(hostname=output.host, agent='Blue', session=session)
    return None

    

if __name__ == "__main__":
    client = OpenAI()
    scenario = 'Scenario2'
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    cyborg = CybORG(path, 'sim', agents={'Red': RedMeanderAgent})
    env = BlueTableWrapper(cyborg)
    
    results = env.reset(agent='Blue')
    obs, reward, done, action_space = results.observation, results.reward, results.done, results.action_space
    obs.del_column("Subnet")
    obs.del_column("IP Address")
    history = f"State: \n{str(obs)}"
    r = []
    blue_action = []
    for i in tqdm(range(30)):
        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        conversation.extend([{"role": "user", "content": HUMAN_PROMPT.format(history=history)}])
        # print("*" * 100, "prompt")
        # print(history)
        response = client.responses.parse(
                model = "gpt-4.1",
                temperature=1.0,
                input=conversation,
                text_format=Action,
            )
        output = response.output_parsed
        action = convert_model_output_to_action(output)
        # print("^" * 100)
        # print(output)
        # print('-' * 40)
        results = env.step(action=action,agent='Blue')
        obs, reward, done, action_space = results.observation, results.reward, results.done, results.action_space
        obs.del_column("Subnet")
        obs.del_column("IP Address")
        # print(results)
        history = f"{history}\nAction: {results.action}\nReward: {results.reward}\nState: {obs}"
        r.append(results.reward)
        blue_action.append(results.action)
        # print()
        # print()
        # print()
    print(r)
    print(blue_action)
    print(sum(r))

