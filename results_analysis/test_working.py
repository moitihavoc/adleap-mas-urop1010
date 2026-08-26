import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from src.envs.LevelForagingEnv import LevelForagingEnv, Agent, Task
from src.envs.StigmergicLevelForagingEnv import StigmergicLevelForagingEnv
from src.utils.math import random_unit_parts
from src.utils.find_visible import find_visible_grids

display = True
dim = (10,10)
visibility = 'partial'
method = 'mcts'
levels = random_unit_parts(3)

components = {
    'agents' : [
            Agent(index='A',atype=method,position=(1,1),direction=1*np.pi/2,radius=0.25,angle=1,level=0.9), # level set to 0.9 to force the agent to collaborate
            Agent(index='1',atype='l1',position=(8,1),direction=1*np.pi/2,radius=0.25,angle=1,level=levels[0]),
            Agent(index='2',atype='l2',position=(1,8),direction=1*np.pi/2,radius=0.25,angle=1,level=levels[1]),
            Agent(index='3',atype='l3',position=(8,9),direction=1*np.pi/2,radius=0.25,angle=1,level=levels[2])
                ],
    'adhoc_agent_index' : 'A',
    'tasks' : [
            Task(index='0',position=(8,8),level=1.0),
            Task(index='1',position=(5,5),level=0.9),
            Task(index='2',position=(0,0),level=0.7),
            Task(index='3',position=(9,1),level=1.0)
                ]
}

base_env = LevelForagingEnv(shape=dim,components=components,display=display) # removed visibility parameter since unavailable
env = StigmergicLevelForagingEnv(base_env, dim=dim[0], decay_rate=0.2)
obs_tensor = env.reset()
state_tensor, reward, done, info = env.step(action=1)

print(state_tensor.shape)  # Should print (9, dim, dim) 
print(state_tensor)
