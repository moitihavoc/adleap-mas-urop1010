from copy import deepcopy
from gymnasium import spaces
import numpy as np
import random 

from src.envs.AdhocReasoningEnv import AdhocAgent, AdhocReasoningEnv, StateSet

MISSHEARD_P = 0.15

"""
    Load Scenario method
"""
def load_default_scenario(method,scenario_id=0,display = False):
    _, scenario_id = load_default_scenario_components(scenario_id)
    components = {"agents":[Agent(index= 0, type= method)]}
    env = TigerEnv(components=components,tiger_pos=random.choice(['left','right']),display=display)  
    return env, scenario_id

def load_default_scenario_components(scenario_id):
    if scenario_id >= 1:
        print('There is no different scenarios for the Tiger problem. Setting scenario_id to 0.')
        scenario_id = 0
    return None, scenario_id

"""
    Support classes
"""
class Agent(AdhocAgent):
    def __init__(self,index,type="random"):
        super(Agent,self).__init__(index,type)
        self.type = type
        self.index = index
    
    def copy(self):
        copy_agent = Agent(self.index,self.type)
        return copy_agent

class TigerEnvState(spaces.Space):
    def __init__(self):
        super().__init__(dtype=str)

###########################################################################
# Helper functions
###########################################################################

def end_condition(env):
    return env.last_action in ['left','right']

def reward(state, next_state):
    return 0

def reward_intermediate(env, action):
    action_name = env.action_dict[action]
    if action_name == "listen":
        return -0.1
    if env.tiger_pos == "left":
        return 1 if action_name == "right" else -10
    return 1 if action_name == "left" else -10

def listen(env):
    if np.random.rand() < 1 - MISSHEARD_P:
        return env.tiger_pos
    return "left" if env.tiger_pos == "right" else "right"

def tiger_transition(action, env):
    info = {}

    # Performing the action
    env.components["agents"][0].next_action = action
    action_name = env.action_dict[action]
    env.last_action = action_name
    if action_name == 'listen':
        env.last_observation = listen(env)
    else:
        env.last_observation = None

    # Calculating the reward
    info["reward"] = reward_intermediate(env, action)

    # Returning the next state and the reward
    return env, info

def environment_transformation(env):
    return env


###########################################################################
# Tiger Environment
###########################################################################

class TigerEnv(AdhocReasoningEnv):

    actions = [0,1,2]

    action_dict = {\
        0:'left',
        1:'right',
        2:'listen'
    }

    observation_dict = {
        0: "left",
        1: "right",
    }

    def __init__(self,components,tiger_pos,display=False):
        self.viewer = None
        self.display = display 

        ###
        # Env settings
        ###
        self.tiger_pos = tiger_pos
        self.last_action = None
        self.last_observation = None
        self.state = {'tiger_pos':tiger_pos}

        state_set = StateSet(TigerEnvState,end_condition=end_condition)
        action_space = spaces.Discrete(3)
        reward_function = reward
        observation_space = environment_transformation
        transition_function = tiger_transition

        ###
        # Initialising the env
        ###
        super(TigerEnv,self).__init__(state_set, transition_function,\
            action_space, reward_function, observation_space, components)
        
        self.state_set.initial_components = self.copy_components(components) 
        self.state_set.initial_state = {'tiger_pos':tiger_pos}

    ######################################################################

    def state_is_equal(self, state):
        return self.state["tiger_pos"] == state.state["tiger_pos"]
    
    def hash_state(self):
        return hash(self.state["tiger_pos"])

    ######################################################################

    def get_observation(self):
        return self.last_observation

    def observation_is_equal(self, obs):
        return self.last_observation == obs

    def hash_observation(self):
        return hash(self.last_observation)

    ######################################################################

    def sample_state(self, agent):
        env = self.copy()
        env.tiger_pos = random.choice(["left", "right"])
        env.state["tiger_pos"] = env.tiger_pos
        env.last_observation = None
        return env

    def sample_nstate(self, agent, n):
        return [self.sample_state(agent) for _ in range(n)]

    ######################################################################

    def get_trans_p(self, action):
        if self.action_dict[action] == "listen":
            return 1.0
        return 0.5

    def get_obs_p(self, action):
        if action is None:
            return 0.5
        if self.action_dict[action] != "listen":
            return 0.5
        if self.last_observation == self.tiger_pos:
            return 1 - MISSHEARD_P
        return MISSHEARD_P

    ######################################################################

    def get_actions_list(self):
        return list(self.action_dict.keys())

    def get_observations_list(self):
        return list(self.observation_dict.values())

    ######################################################################

    def get_feature(self):
        return np.array([0.5, 0.5])

    ######################################################################

    def get_adhoc_agent(self):
        return self.components["agents"][0]

    ######################################################################

    def copy(self):
        components = self.copy_components(self.components)

        env = TigerEnv(
            components,
            deepcopy(self.tiger_pos),
            self.display,
        )

        env.viewer = self.viewer
        env.episode = self.episode
        env.simulation = self.simulation

        env.state = deepcopy(self.state)
        env.last_observation = deepcopy(self.last_observation)

        env.state_set.initial_state = deepcopy(
            self.state_set.initial_state
        )

        return env

    ######################################################################

    def import_method(self, agent_type):
        from importlib import import_module
        try:
            module = import_module('src.reasoning.'+agent_type)
        except:
            raise NotImplemented

        method = getattr(module, agent_type+'_planning')
        return method

    def render(self,mode='human',sleep_=0.5):
        return False
    