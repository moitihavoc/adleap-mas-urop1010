###
# IMPORTS
###
import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from src.envs.LevelForagingEnv import LevelForagingEnv, Agent, Task
from src.utils.math import random_unit_parts

###
# Setting the environment
###
display = True
dim = (10,10)
visibility = 'partial'
method = 'mcts'

# Randomly split 1.0 into 4 parts for agent level initialization
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

env = LevelForagingEnv(shape=dim,components=components,display=display) # removed visibility parameter since unavailable

###
# ADLEAP-MAS MAIN ROUTINE
###
state = env.reset()
print(state.get_observation())

done, max_episode = False, 200
while env.episode < max_episode and not done:
    print('|||| Episode',env.episode)
    # 1. Importing agent method
    adhoc_agent = env.get_adhoc_agent()
    print("Ad hoc agent:", adhoc_agent)
    print("Ad hoc agent type:", adhoc_agent.type)
    method = env.import_method(adhoc_agent.type)

    # 2. Reasoning about next action and target
    action, target = method(state, adhoc_agent)

    # 3. Taking a step in the environment
    state, reward, done, info = env.step(action)
    print(state.state,  action)
    print("Ad hoc agent's explicit observation:", env.get_observation())
    #adhoc_agent.show_memory()

env.close()
###
# THE END - That's all folks :)
###
