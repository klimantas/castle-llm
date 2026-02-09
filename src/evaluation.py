"""Episode evaluation and result collection."""

import json
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import List, Dict, Any

from .agent import CyberDefenseAgent
from .config import RESULTS_DIR


class EvaluationRunner:
    """Runner for evaluating agents over multiple episodes."""
    
    def __init__(
        self,
        agent: CyberDefenseAgent,
        experiment_version: str = "v0-0-1"
    ):
        """Initialize the evaluation runner.
        
        Args:
            agent: The cyber defense agent to evaluate.
            experiment_version: Version string for this experiment.
        """
        self.agent = agent
        self.experiment_version = experiment_version
        
        # Create experiment directory
        year_month = datetime.now().strftime("%Y%m")
        self.experiment_dir = RESULTS_DIR / f"results_{experiment_version}_{year_month}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    def run_evaluation(
        self,
        model_name: str,
        temperature: float,
        red_agent_name: str,
        episodes: int = 3,
        steps_per_episode: int = 30
    ) -> Dict[str, Any]:
        """Run evaluation for multiple episodes.
        
        Args:
            model_name: Name of the model being evaluated.
            temperature: Temperature setting used.
            red_agent_name: Name of the red agent.
            episodes: Number of episodes to run.
            steps_per_episode: Steps per episode.
            
        Returns:
            Dictionary containing evaluation results.
        """
        all_rewards = []
        all_impacts = []
        episode_results = []
        
        print(f"\n{'='*60}")
        print(f"Running {episodes} episodes of {model_name}")
        print(f"{'='*60}\n")
        
        for episode in range(episodes):
            print(f"\n{'─'*60}")
            print(f"Episode {episode + 1}/{episodes}")
            print(f"{'─'*60}")
            
            (
                total_reward,
                step_rewards,
                actions,
                red_actions,
                impacts,
                api_failures,
                parsing_failures
            ) = self.agent.run_episode(steps=steps_per_episode)
            
            all_rewards.append(total_reward)
            all_impacts.append(impacts)
            
            episode_results.append({
                "episode_number": episode + 1,
                "total_reward": total_reward,
                "step_rewards": step_rewards,
                "actions": [str(a) for a in actions],
                "red_actions": [str(a) for a in red_actions],
                "impacts": impacts,
                "api_failures": api_failures,
                "parsing_failures": parsing_failures,
                "api_failure_count": len(api_failures),
                "parsing_failure_count": len(parsing_failures),
                "failure_rate": (len(api_failures) + len(parsing_failures)) / steps_per_episode,
            })
            
            print(f"\n✓ Episode {episode + 1} completed")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  API Failures: {len(api_failures)}/{steps_per_episode}")
            print(f"  Parsing Failures: {len(parsing_failures)}/{steps_per_episode}")
        
        # Calculate statistics
        statistics = {
            "mean_reward": mean(all_rewards),
            "stdev_reward": stdev(all_rewards) if len(all_rewards) > 1 else None,
            "min_reward": min(all_rewards),
            "max_reward": max(all_rewards),
            "all_total_rewards": all_rewards,
            "mean_impacts": mean(all_impacts),
        }
        
        # Prepare results log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_log = {
            "timestamp": timestamp,
            "scenario": self.agent.scenario,
            "episodes": episodes,
            "steps_per_episode": steps_per_episode,
            "model": model_name,
            "temperature": temperature,
            "red_agent": red_agent_name,
            "episodes": episode_results,
            "statistics": statistics,
        }
        
        # Print summary
        self._print_summary(statistics)
        
        # Save results
        filename = self.experiment_dir / f"{timestamp}_{model_name.replace('.', '_')}.json"
        with open(filename, "w") as f:
            json.dump(results_log, f, indent=2)
        
        print(f"\n✓ Results saved to: {filename}")
        print(f"{'='*60}\n")
        
        return results_log
    
    def _print_summary(self, statistics: Dict[str, Any]) -> None:
        """Print evaluation summary."""
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"All Total Rewards: {statistics['all_total_rewards']}")
        print(f"Average Reward: {statistics['mean_reward']:.2f}")
        if statistics['stdev_reward'] is not None:
            print(f"Std Dev: {statistics['stdev_reward']:.2f}")
        print(f"Min Reward: {statistics['min_reward']:.2f}")
        print(f"Max Reward: {statistics['max_reward']:.2f}")
        print(f"Average Impacts: {statistics['mean_impacts']:.2f}")
