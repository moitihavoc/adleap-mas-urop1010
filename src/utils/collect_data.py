import torch
import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from src.envs.LevelForagingEnv import LevelForagingEnv, Agent, Task, generate_random_estimation_scenario
from src.envs.StigmergicLevelForagingEnv import StigmergicLevelForagingEnv
from src.utils.math import random_unit_parts
from src.utils.tensor_convert import dict_to_tensor
from src.utils.find_visible import find_visible_grids
from src.communication.traces import TraceField
from src.reasoning.levelbased.l1 import l1_planning


def fixed_trace_decay(traces):
    """
    Work around for TraceField.decay() indexing bug.
    Decays intensity, increments age, and zeroes out all channels
    where intensity has dropped to zero.
    """
    traces.fields[0] = np.maximum(0.0, traces.fields[0] - traces.decay_rate)
    traces.fields[1] += 1.0
    # Zero out all channels where intensity (channel 0) is zero
    zero_mask = traces.fields[0] <= 0
    for ch in range(traces.fields.shape[0]):
        traces.fields[ch][zero_mask] = 0.0


def collect_episode(dim, nagents, ntasks):
    """
    Collect a single episode of data using the l1 heuristic policy.
    Returns a list of dicts, each containing:
        - obs: (1, 9, dim, dim) tensor observation
        - action: int action taken
        - reward: float reward received
        - pi: (5,) numpy array, one-hot policy prior
        - value: float, discounted return (filled retroactively)
    """
    # 1. Create a fresh random environment
    env, _ = generate_random_estimation_scenario(
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

    # 2. Reset the environment and initialise trace field
    state = env.reset()
    traces = TraceField(dim=dim)

    # Build initial observation tensor
    # dict_to_tensor expects:
    #   - 'agents': list of objects with .position attribute (Agent objects)
    #   - 'tasks': list of [index, x, y] entries
    #   - 'obstacles': list of objects with .position attribute or raw tuples
    adhoc_agent = env.get_adhoc_agent()
    observable_env = env.observation_space(env.copy())
    obs_dict = {
        'agents': observable_env.components['agents'],
        'tasks': [[t.index, t.position[0], t.position[1]]
                  for t in observable_env.components['tasks'] if not t.completed],
        'obstacles': observable_env.components['obstacles']
    }
    visible_grids = find_visible_grids(env)
    obs_tensor = dict_to_tensor(obs_dict, traces.fields, dim, visible_grids)

    episode_data = []
    done = False
    max_steps = 200
    step = 0

    while not done and step < max_steps:
        # 3. Get the observable state for the heuristic to reason over
        observable_state = env.observation_space(env.copy())
        adhoc_agent = env.get_adhoc_agent()

        # 4. Use the l1 heuristic to choose an action
        action, target = l1_planning(observable_state, adhoc_agent)
        adhoc_agent.target = target

        # 5. Record current observation and one-hot policy prior
        current_obs = obs_tensor.clone()
        pi_t = np.zeros(5, dtype=np.float32)
        pi_t[action] = 1.0

        # 6. Step the environment
        state, reward, done, info = env.step(action)

        # 7. Decay traces and emit new ones for each agent
        fixed_trace_decay(traces)
        for agent in env.components['agents']:
            level = agent.level if agent.level is not None else 0.0
            vis_comps = env.get_visible_components(agent)
            # vis_comps is a dict with keys: 'agents', 'tasks', 'obstacles'
            # each task entry is [index, x, y]
            visible_tasks = vis_comps['tasks']
            if visible_tasks:
                task_level_sum = 0.0
                for tk in env.components['tasks']:
                    if not tk.completed and [tk.index, tk.position[0], tk.position[1]] in visible_tasks:
                        task_level_sum = max(task_level_sum, tk.level)
                help_signal = 1 if level < task_level_sum else 0
            else:
                help_signal = 0

            claim = 1 if any(
                tk.completed for tk in env.components['tasks']
                if [tk.index, tk.position[0], tk.position[1]] in visible_tasks
            ) else 0 if visible_tasks else 0

            traces.emit(level, help_signal, claim, agent.position)

        # 8. Build next observation tensor from the observable state
        observable_env = env.observation_space(env.copy())
        obs_dict = {
            'agents': observable_env.components['agents'],
            'tasks': [[t.index, t.position[0], t.position[1]]
                      for t in observable_env.components['tasks'] if not t.completed],
            'obstacles': observable_env.components['obstacles']
        }
        visible_grids = find_visible_grids(env)
        obs_tensor = dict_to_tensor(obs_dict, traces.fields, dim, visible_grids)

        # 9. Record the step
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
    collect_data(num_episodes=300, save_path="src/Training_Data/training_data.pt")
