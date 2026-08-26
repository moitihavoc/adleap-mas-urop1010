import torch
import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from src.envs.LevelForagingEnv import LevelForagingEnv, Agent, Task, generate_random_estimation_scenario
from src.envs.StigmergicLevelForagingEnv import StigmergicLevelForagingEnv
from src.utils.math import random_unit_parts
from src.reasoning.levelbased.l1 import l1_planning


def collect_episode(dim, nagents, ntasks):
    """
    Collect a single episode of data using src.reasoning.levelbased.l1
    During training, the ad hoc agent has full observability 
    Returns a list of dicts, each containing:
        - obs: (9, dim, dim) tensor observation
        - action: int 
        - reward: float 
        - pi: (5,) numpy array, one-hot policy prior
        - value: float, discounted return 
    """
    # Create a fresh random environment
    base_env, _ = generate_random_estimation_scenario(
        method='l1',
        adhoc_pos=(1, 1),
        dim=(dim, dim),
        nagents=nagents,
        ntasks=ntasks,
        type_knowledge=False,
        parameter_knowledge=True,
        vision_block=False,
        template_types=['l1', 'l2'],
        parameters_minmax=[0.5, 1.0],
        seed=None,
        display=False
    )
    base_env.get_adhoc_agent().level = 0.9
    env = StigmergicLevelForagingEnv(base_env, dim=dim)

    # Reset the wrapped environment and obtain the initial tensor observation.
    obs_tensor = env.reset()

    episode_data = []
    done = False
    max_steps = 200
    step = 0

    while not done and step < max_steps:
        # Get the observable state for the heuristic to reason over
        adhoc_agent = base_env.get_adhoc_agent()
        observable_state = base_env.observation_space(base_env.copy())

        # Use the l1 heuristic to choose an action
        action, target = l1_planning(observable_state, adhoc_agent)
        adhoc_agent.target = target

        # Record current observation and one-hot policy prior
        current_obs = obs_tensor.clone()
        pi_t = np.zeros(5, dtype=np.float32)
        pi_t[action] = 1.0
        obs_tensor, reward, done, info = env.step(action)

        # Record the step
        episode_data.append({
            'obs': current_obs,
            'action': action,
            'reward': reward,
            'pi': pi_t,
            'value': 0.0  # placeholder, filled retroactively
        })

        step += 1

    return episode_data


def compute_discounted_returns(episode_data, gamma=0.99):
    """Retroactively compute discounted returns for each step."""
    R = 0.0
    for i in reversed(range(len(episode_data))):
        R = episode_data[i]['reward'] + gamma * R
        episode_data[i]['value'] = R
    return episode_data


def collect_data(num_episodes=300, dim=10, nagents=2, ntasks=5,
                 gamma=0.99, save_path="training_data.pt"):
    """
    Collect training data by running episodes with the l1 heuristic policy.
    Saves a list of episodes, where each episode is a list of step dicts.
    """
    print(f"Starting data collection for {num_episodes} episodes...")
    print(f"  Grid: {dim}x{dim} | Agents: {nagents} | Tasks: {ntasks}")
    all_episodes = []
    total_rewards = 0
    total_steps = 0

    for ep in range(num_episodes):
        # Collect one episode
        episode_data = collect_episode(dim, nagents, ntasks)

        # Compute discounted returns
        episode_data = compute_discounted_returns(episode_data, gamma)

        # Track stats
        ep_reward = sum(step['reward'] for step in episode_data)
        total_rewards += ep_reward
        total_steps += len(episode_data)

        all_episodes.append(episode_data)

        if (ep + 1) % 10 == 0:
            avg_reward = total_rewards / (ep + 1)
            avg_steps = total_steps / (ep + 1)
            print(f"  [{ep + 1}/{num_episodes}] "
                  f"avg_reward={avg_reward:.2f} avg_steps={avg_steps:.1f}")

    # Save the dataset
    torch.save(all_episodes, save_path)
    print(f"\nData collection complete!")
    print(f"  Episodes: {num_episodes}")
    print(f"  Total steps: {total_steps}")
    print(f"  Avg reward/episode: {total_rewards / num_episodes:.2f}")
    print(f"  Avg steps/episode: {total_steps / num_episodes:.1f}")
    print(f"  Saved to: {save_path}")


if __name__ == "__main__":
    collect_data(num_episodes=300, save_path="src/Training_Data/training_data_l1.pt")
