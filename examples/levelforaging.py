###
# IMPORTS
###
import sys
import os
sys.path.append(os.getcwd())

from src.envs.LevelForagingEnv import load_default_scenario

###
# Setting the environment
###
display = True
dim = (10,10)
visibility = 'partial'
method = 'pomcp'

scenario_id = 0
env, scenario_id = load_default_scenario(method,scenario_id,display=display)

###
# ADLEAP-MAS MAIN ROUTINE
###
state = env.reset()

done, max_episode = False, 200
while env.episode < max_episode and not done:
    print('|||| Episode',env.episode)
    # 1. Importing agent method
    adhoc_agent = env.get_adhoc_agent()
    method = env.import_method(adhoc_agent.type)

    # 2. Reasoning about next action and target
    action, target = method(state, adhoc_agent)

    # 3. Taking a step in the environment
    state, reward, done, info = env.step(action)

    # if you want to visualize the ad hoc agent memory abou the environment,
    # remove the bellow comment to print it 
    #adhoc_agent.show_memory()

env.close()
###
# THE END - That's all folks :)
###
