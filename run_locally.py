##
## Template to run multiple experiments on AdLeap-MAS
##
# 1. Setting the environment
atype = 'pomcp'                # choose your method
kwargs = {}                     # define your additional hyperparameters to it (optional)
env_name = 'LevelForagingEnv'   # choose your environment
scenario_id = 5                 # define your scenario configuration (check the available configuration in our GitHub)

display = False                 # choosing to turn on or off the display
NEXP = 50

####
# Imports
###
import sys
import os
import time

sys.path.append(os.getcwd())

from src.log import LogFile
from src.envs.LevelForagingEnv import load_default_scenario, is_mas

for exp_num in range(NEXP):
    print(f'Running experiment {exp_num+1}/{NEXP}...')
    ###
    # Setting the environment
    ###
    env, scenario_id = load_default_scenario(atype,scenario_id=scenario_id,display=False)

    ###
    # ADLEAP-MAS MAIN ROUTINE
    ###
    state = env.reset()
    agent = env.get_adhoc_agent()

    if is_mas(scenario_id):
        header = ['Iteration','Reward','Time to reason','N Rollouts', 'N Simulations',
                'Coop?']#,'Type Belief']
    else:
        header = ['Iteration','Reward','Time to reason','N Rollouts', 'N Simulations']
    log = LogFile('LevelForagingEnv',scenario_id,atype,exp_num,header)

    MAX_EPISODES = 200
    done = False
    while not done and env.episode < MAX_EPISODES:
        print(f'Episode {env.episode}')
        # 1. Importing agent method
        method = env.import_method(agent.type)

        # 2. Reasoning about next action and target
        start = time.time()
        agent.next_action, _ = method(state, agent)
        end = time.time()
        
        # 3. Taking a step in the environment
        state,reward,done,info = env.step(agent.next_action)
        
        #if is_mas(scenario_id):
        #    info['typebelief'] = agent.get_type_belief()

        data = {'it':env.episode,
                'reward':reward,
                'time':end-start,
                'nrollout':agent.smart_parameters['count']['nrollouts'],
                'nsimulation':agent.smart_parameters['count']['nsimulations'],
                'coop':info['coop']}#, 'typebelief':info['typebelief']}
        log.write(data)

    env.close()
    ###
    # THE END - That's all folks :)
    ###