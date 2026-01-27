###
# Imports
###
import sys
import os
sys.path.append(os.getcwd())

from src.envs.RockSampleEnv import load_default_scenario

###
# Setting the environment
###
display = True
scenario_id = 0
method = 'ibpomcp'

env, scenario_id = load_default_scenario(method,scenario_id,display=display)

"""
### Components templates ###
# - if you want to run a custom scenario, remove this comment block, modify the
# scenario components and above settings as you want

start_pos = (3,3)
agent = Agent(0, start_pos, method)
components = {"agents":[agent],
    "rocks": [  Rock(0, (2,2),"Good"),
                Rock(1, (5,2),"Good"),
                Rock(2, (3,7), "Bad"),
                Rock(3, (6,8), "Bad")]}

dim = [10,10]
display = True
env = RockSampleEnv(components=components,dim=dim,display=display)
"""

###
# ADLEAP-MAS MAIN ROUTINE
###
state = env.reset()
agent = env.get_adhoc_agent()

done = False
while env.episode < 200 and not done:
    # 1. Importing agent method
    agent = env.get_adhoc_agent()
    method = env.import_method(agent.type)

    # 2. Reasoning about next action and target
    agent.next_action, _ = method(state, agent)

    # 3. Taking a step in the environment
    state,_,done,_ = env.step(action=agent.next_action)

env.close()
###
# THE END - That's all folks :)
###