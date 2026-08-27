import random

import numpy as np
import torch

from src.envs.LevelForagingEnv import generate_random_estimation_scenario
from src.envs.StigmergicLevelForagingEnv import StigmergicLevelForagingEnv


NUM_SCENARIOS = 100
DIM = 10
NAGENTS = 2
NTASKS = 5
ADHOC_RADIUS = 0.25
ADHOC_ANGLE = 1.0
ADHOC_LEVEL = 0.9


def create_evaluation_set(num_scenarios=NUM_SCENARIOS, seed=2026):
    """Create one shared evaluation set with the training distribution."""
    random.seed(seed)
    scenarios = []

    for _ in range(num_scenarios):
        base_env, _ = generate_random_estimation_scenario(
            method='l1', adhoc_pos=(1, 1), dim=(DIM, DIM),
            nagents=NAGENTS, ntasks=NTASKS, type_knowledge=False,
            parameter_knowledge=True, vision_block=False,
            template_types=['l1', 'l2'], parameters_minmax=[0.5, 1.0],
            seed=None, display=False
        )
        base_env.reset()

        adhoc_agent = base_env.get_adhoc_agent()
        adhoc_agent.radius = ADHOC_RADIUS
        adhoc_agent.angle = ADHOC_ANGLE
        adhoc_agent.level = ADHOC_LEVEL
        scenarios.append(base_env)

    return scenarios


def run_evaluation(model, scenarios, ablate=""):
    """Evaluate a model on shared scenario copies and return per-scenario data."""
    rewards = []
    steps = []
    successes = []

    for base_env in scenarios:
        env = StigmergicLevelForagingEnv(base_env.copy(), dim=DIM)
        obs_tensor = env.reset()
        done = False
        step = 0
        episode_reward = 0

        while not done and step < 200:
            input_tensor = obs_tensor.unsqueeze(0).clone()
            if ablate == "no trace":
                input_tensor[:, 3:8, :, :] = 0.0
            elif ablate == "basic":
                input_tensor[:, 5:8, :, :] = 0.0
            elif ablate == "level":
                input_tensor[:, 6:8, :, :] = 0.0

            with torch.no_grad():
                pi_logits, _ = model(input_tensor)
                action = torch.argmax(pi_logits, dim=1).item()

            obs_tensor, reward, done, _ = env.step(action)
            episode_reward += reward
            step += 1

        rewards.append(episode_reward)
        steps.append(step)
        successes.append(episode_reward > 0)

    return {
        'rewards': rewards,
        'steps': steps,
        'successes': successes,
    }


def summarize(values):
    """Return mean and a 95% normal-approximation confidence-interval margin."""
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return mean, 1.96 * standard_error
