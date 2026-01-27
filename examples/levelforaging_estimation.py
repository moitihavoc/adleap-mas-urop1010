###
# IMPORTS
###
import time
import sys
import os
sys.path.append(os.getcwd())

from src.log import EstimationLogFile
from src.utils.args import get_estimation_args
from src.envs.LevelForagingEnv import load_default_scenario, generate_random_estimation_scenario

###
# Setting the environment
###
args = get_estimation_args()
estimation_method = args.mode

###
# ADLEAP-MAS MAIN ROUTINE
###
if args.id == 'random':
    env, scenario_id = generate_random_estimation_scenario(args.atype,adhoc_pos=(1,1),
        dim=(30,30),nagents=args.nagents,ntasks=args.ntasks,type_knowledge=False,\
        parameter_knowledge=True,vision_block=False,template_types=['l1','l2'],\
        parameters_minmax=[0.5,1.],seed=24,display=False)
else:
    env, scenario_id = load_default_scenario(args.atype,int(args.id),display=False)

adhoc_agent = env.get_adhoc_agent()
true_types, true_parameters = env.get_agents_type_n_parameters()

state = env.reset()

if estimation_method.upper() == 'BAE' or estimation_method.upper() == 'OEATE_A':
    template_types = env.components['template_types']
    if 'adversary' in template_types:
        template_types.remove('adversary')
        env.state_set.initial_components = env.copy_components(env.components)
    state = env.reset()
        
    estimation_kwargs = { 
        'template_types':template_types,\
        'parameters_minmax':[(0.5,1),(0.5,1),(0.5,1)],
        'adversary_last_action':None
        }
    
    if estimation_method.upper() == 'BAE':
        from src.reasoning.estmethods import bae
        supmethod = bae.BAE(env,estimation_kwargs['template_types'],\
                    estimation_kwargs['parameters_minmax']).estimation_method_name
        if supmethod is None:
            method_name = estimation_method
        else:
            method_name = estimation_method+'_'+supmethod
    else:
        method_name = estimation_method
else:
    template_types = env.components['template_types']
    estimation_kwargs = { 
        'template_types':template_types,\
        'parameters_minmax':[(0.25,1),(0.25,1),(0.25,1)],
        'adversary_last_action':None
        }
    
    method_name = estimation_method

log = EstimationLogFile('LevelForagingEnv',scenario_id,args.atype,method_name,args.exp_num,\
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
    if env.is_adversarial():
        adhoc_agent.smart_parameters['estimation_method'] = estimation_method
        adhoc_agent.smart_parameters['estimation_args'] = estimation_kwargs
        action, target = method(state, adhoc_agent, adversary = True, mode='max')
    # Foraging problems
    else:
        adhoc_agent.smart_parameters['estimation_method'] = estimation_method
        adhoc_agent.smart_parameters['estimation_args'] = estimation_kwargs
        action, target = method(state, adhoc_agent)
    end = time.time()
    memory_usage = 'NotImplemented' #adhoc_agent.smart_parameters['search_tree'].size_in_memory()

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
    
    i, typeestimation_err, parametersestimation_err = 0, [], []
    for agent in env.components['agents']:
        if agent.index != adhoc_agent.index:
            true_type_index = template_types.index(true_types[agent.index])
            typeestimation_err.append(1 - \
                    typeestimation[i][true_type_index])
            
            parametersestimation_err.append([
                abs(true_parameters[agent.index][0] - parametersestimation[i][true_type_index][0]),
                abs(true_parameters[agent.index][1] - parametersestimation[i][true_type_index][1]),
                abs(true_parameters[agent.index][2] - parametersestimation[i][true_type_index][2]),
                ])
            i += 1

    data = {'it':env.episode,
            'reward':reward,
            'time':end-start,
            'nrollout':adhoc_agent.smart_parameters['count']['nrollouts'],
            'nsimulation':adhoc_agent.smart_parameters['count']['nsimulations'],
            'typeestimation':typeestimation,
            'typeestimation_err':typeestimation_err,
            'parametersestimation':parametersestimation,
            'parametersestimation_err':parametersestimation_err,
            'memoryusage':memory_usage,}
    log.write(data)

    if done:
        env.respawn_tasks()

env.close()
###
# THE END - That's all folks :)
###
