import os
import sys

sys.path.append(os.getcwd())

from src.envs.AdhocReasoningEnv import AdhocAgent
from src.envs.ChessEnv import ChessEnv

components = {  'white':AdhocAgent('white','mcts'),
                'black':AdhocAgent('black','pomcp')  }

env = ChessEnv(components,display=True)

state = env.reset()

done = False
while not done:
    adhoc_agent = env.get_adhoc_agent()
    print(adhoc_agent.index)
    method = env.import_method(adhoc_agent.type)
    if state.current_player == 'white':
        adhoc_agent.next_action, _ = method(state, adhoc_agent, adversary=True, target='max')
    else:
        adhoc_agent.next_action, _ = method(state, adhoc_agent, adversary=True, target='min')
    state,_,done,_ = env.step(adhoc_agent.next_action)
    print(state.state,'\n\n')
env.close()