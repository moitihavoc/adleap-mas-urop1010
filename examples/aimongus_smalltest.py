###
# IMPORTS
###
import time
import sys
import os
sys.path.append(os.getcwd())

from src.log import EstimationLogFile
from src.utils.args import get_estimation_args
from src.envs.AImongUsEnv import load_default_scenario

###
# Setting the environment
###
args = get_estimation_args()

estimation_method = args.mode
template_types = ['l1','l2','l3','l4']
estimation_kwargs = { 
    'template_types':template_types,\
    'parameters_minmax':[(0.5,1),(0.5,1),(0.5,1)],
    'adversary_last_action':None
    }

###
# ADLEAP-MAS MAIN ROUTINE
###
env, scenario_id = load_default_scenario(args.atype,args.id,display=True)
state = env.reset()

log = EstimationLogFile('LevelForagingEnv',scenario_id,args.atype,estimation_method,args.exp_num,\
    estimation_kwargs['template_types'],estimation_kwargs['parameters_minmax'])

done, max_episode = False, 200
while env.episode < max_episode:
    #print('|||| Episode',env.episode)
    # 1. Importing agent method
    adhoc_agent = env.get_adhoc_agent()
    method = env.import_method(adhoc_agent.type)

    # 2. Reasoning about next action and target
    # Adversarial problems
    start = time.time()
    adhoc_agent.smart_parameters['estimation_method'] = estimation_method
    adhoc_agent.smart_parameters['estimation_kwargs'] = estimation_kwargs
    action, target = method(state, adhoc_agent, adversary = True, mode='max')
    end = time.time()
    memory_usage = adhoc_agent.smart_parameters['search_tree'].size_in_memory()

    # 3. Taking a step in the environment
    state, reward, done, info = env.step(action)

    # if you want to visualize the ad hoc agent memory abou the environment,
    # remove the bellow comment to print it 
    #adhoc_agent.show_memory()

    # if you want to visualize the ad hoc agent estimation about the environment,
    # remove the bellow comment to print it 
    #adhoc_agent.smart_parameters['estimation'].show_estimation(env)

    typeestimation, parametersestimation, _ =\
        adhoc_agent.smart_parameters['estimation'].get_estimation(env)
    data = {'it':env.episode,
            'reward':reward,
            'time':end-start,
            'nrollout':adhoc_agent.smart_parameters['count']['nrollouts'],
            'nsimulation':adhoc_agent.smart_parameters['count']['nsimulations'],
            'typeestimation':typeestimation,
            'parametersestimation':parametersestimation,
            'memoryusage':memory_usage,}
    log.write(data)

    if done:
        env.respawn_tasks()

env.close()
###
# THE END - That's all folks :)
###
