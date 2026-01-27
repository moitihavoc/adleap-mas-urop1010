###
# Imports
###
import sys
import os
import time
sys.path.append(os.getcwd())

from src.log import LogFile
from src.utils.args import get_args
from src.envs.AdhocReasoningEnv import AdhocAgent
from src.envs.ChessEnv import ChessEnv

###
# Setting the environment
###
args = get_args()
components = {  'white':AdhocAgent('white','mcts'),
                'black':AdhocAgent('black','pomcp')  }
env = ChessEnv(components,display=False)

###
# ADLEAP-MAS MAIN ROUTINE
###
state = env.reset()
adhoc_agent = env.get_adhoc_agent()

header = ['Iteration','Reward','Time to reason','N Rollouts', 'N Simulations']
log = LogFile('ChessEnv',0,args.atype,args.exp_num,header)


done = False
while not done:
    # 1. Importing agent method
    adhoc_agent = env.get_adhoc_agent()
    method = env.import_method(adhoc_agent.type)

    # 2. Reasoning about next action and target
    adversary, target = True, 'max' if state.current_player == 'white' else 'min'
    start = time.time()
    adhoc_agent.next_action, _ = method(state, adhoc_agent,adversary=True, target=target)
    end = time.time()

    # 3. Taking a step in the environment
    state,reward,done,_ = env.step(adhoc_agent.next_action)
    data = {'it':env.episode,
            'reward':reward,
            'time':end-start,
            'nrollout':adhoc_agent.smart_parameters['count']['nrollouts'],
            'nsimulation':adhoc_agent.smart_parameters['count']['nsimulations']}
    log.write(data)

env.close()
###
# THE END - That's all folks :)
###
