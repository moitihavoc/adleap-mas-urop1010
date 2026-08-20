import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.envs.LevelForagingEnv import LevelForagingEnv, Agent, Task
from src.communication.traces import TraceField
from src.utils.tensor_convert import dict_to_tensor
from src.utils.find_visible import find_visible_grids

# --- Wrapper for the LevelForagingEnv to include stigmergic communication

"""
What this wapper does:
- Redefines the observation space to include the trace field information.
- Modify the reset and step methods to update the trace field based on agent actions and task states.
"""

class StigmergicLevelForagingEnv(gym.Wrapper):
    def __init__(self, env: LevelForagingEnv, dim:int, decay_rate: float = 0.2):
        super().__init__(env)
        self.dim = dim
        self.decay = decay_rate
        self.traces = TraceField(dim=dim, decay_rate=decay_rate)
        # redefine the observation space to include the trace field information
        self.observation_space = spaces.Box(
            low = 0.0, high = 1.0,
            shape = (9, dim, dim), # 9 channels: 3 for agents, 5 for traces, 1 for visibility
            dtype = np.float32
        )

    def reset(self, **kwargs):
        obs_dict = self.env.reset(**kwargs).get_observation()
        self.traces.reset()
        obs_tensor = dict_to_tensor(obs_dict, self.traces.fields, self.dim)
        return obs_tensor

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        obs_dict = state.get_observation()
        self.traces.decay()

        # construct a tensor based on observable components
        agent = self.env.get_adhoc_agent()
        vb = self.env.get_component('vision_block')
        visible_grids = find_visible_grids(agent, vb)
        obs_tensor = dict_to_tensor(obs_dict, self.traces.fields, self.dim, visible_grids)

        # emit traces for every agent in the environment
        for agent in self.env.components['agents']:
            level = agent.level
            obs_state = self.env.get_visible_components(agent) # get the observable state for each agent
            task_level = obs_state.components['tasks'][0].level if obs_state.components['tasks'] else 0.0
            help_signal = 1 if level < task_level else 0
            claim = 1 if (task.completed for task in obs_state.components['tasks']) else 0
            self.traces.emit(level, help_signal, claim, agent.position)

        return obs_tensor, reward, done, info