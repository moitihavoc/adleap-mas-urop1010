###
# Imports
###
import sys
import os
import time

sys.path.append(os.getcwd())

from src.log import LogFile
from src.utils.args import get_args
from src.envs.TigerEnv import load_default_scenario


###
# TIGER ENVIRONMENT SETTINGS
###
args = get_args()
header = ['Iteration','Reward','Time to reason','N Rollouts', 'N Simulations']
log = LogFile('TigerEnv',0,args.atype,args.exp_num,header)

MAX_EPISODES = 200
###
# ADLEAP-MAS MAIN ROUTINE
###
total_episode = 0
while total_episode < MAX_EPISODES:
    done = False
    env, scenario_id = load_default_scenario(args.atype,0)
    
    state = env.reset()
    agent = env.get_adhoc_agent()
    while total_episode < MAX_EPISODES and not done:
        # 1. Importing agent method
        method = env.import_method(agent.type)

        # 2. Reasoning about next action and target
        start = time.time()
        agent.next_action, _ = method(state, agent)
        end = time.time()

        # 3. Taking a step in the environment
        print('Action:',state.action_dict[agent.next_action])
        next_state, reward, done, _ = env.step(action=agent.next_action)

        data = {'it':env.episode,
                'reward':reward,
                'time':end-start,
                'nrollout':agent.smart_parameters['count']['nrollouts'],
                'nsimulation':agent.smart_parameters['count']['nsimulations']}
        log.write(data)
        state = next_state
        total_episode += 1
    print('Episode:',total_episode,'/',MAX_EPISODES,'-> Reward:',reward)
    env.close()
###
# THE END - That's all folks :)
###