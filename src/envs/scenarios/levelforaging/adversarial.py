import numpy as np
from src.envs.LevelForagingEnv import Agent, Task

S5X5_2AGENTS_1TASKS_1ADV = {
    # Scenario 0: FO Adversarial - hand designed scenario: hard to accomplish
    # but good to evaluate the adversary behaviour
    'dim': (5,5),
    'type_knowledge': False,
    'template_types':['l6','adversary'],
    'parameter_knowledge': False,
    'vision_block': True,
    'agents' : [
        Agent(index='A',atype=None,position=(0,1),
                direction=1*np.pi/2,radius=1.,angle=1.,level=1.), 
        Agent(index='1',atype='l1',position=(1,0),
                direction=0*np.pi/2,radius=1.,angle=1.,level=1.), 
        Agent(index='X',atype='mcts_min',position=(3,0),
                direction=1*np.pi/2,radius=1.,angle=1.,level=1.), 
            ],
    'adhoc_agent_index' : 'A',
    'impostor_index': 'X',
    'tasks' : [
        Task(index='0',position=(4,4),level=1.),
            ],
    'obstacles':[]
}

S9X9_2AGENTS_5TASKS_1ADV = {
    # Scenario 1: FO Adversarial - warehouse benchmark scenario with 
    # 1 adhoc agent 1 teammate 1 adversary and 5 tasks
    'dim': (9,9),
    'type_knowledge': False,
    'template_types':['l6','adversary'],
    'parameter_knowledge': False,
    'vision_block': True,
    'agents' : [
        Agent(index='A',atype=None,position=(0,0),
                direction=1*np.pi/2,radius=1.,angle=1.,level=1.), 
        Agent(index='1',atype='l3',position=(0,8),
                direction=0*np.pi/2,radius=1.,angle=1.,level=1.), 
        Agent(index='X',atype='mcts_min',position=(8,0),
                direction=1*np.pi/2,radius=1.,angle=1.,level=1.), 
            ],
    'adhoc_agent_index' : 'A',
    'impostor_index': 'X',
    'tasks' : [
        Task(index='0',position=(3,3),level=0.5),
        Task(index='1',position=(7,7),level=0.5),
        Task(index='2',position=(7,3),level=0.5),
        Task(index='3',position=(3,7),level=0.5),
        Task(index='4',position=(5,5),level=0.5),
            ],
    'obstacles':[]
}

S9X9_4AGENTS_5TASKS_1ADV = {
    # Scenario 2: FO Adversarial  - warehouse benchmark scenario with 
    # 1 adhoc agent 3 teammate 1 adversary and 5 tasks
    'dim': (9,9),
    'type_knowledge': False,
    'template_types':['l4','l5','l6','adversary'],
    'parameter_knowledge': False,
    'vision_block': True,
    'agents' : [
        Agent(index='A',atype=None,position=(0,0),
                direction=1*np.pi/2,radius=1.,angle=1.,level=1.0), 
        Agent(index='1',atype='l1',position=(0,4),
                direction=0*np.pi/2,radius=1.,angle=1.,level=1.0),  
        Agent(index='2',atype='l2',position=(0,8),
                direction=0*np.pi/2,radius=1.,angle=1.,level=1.0),  
        Agent(index='3',atype='l3',position=(4,0),
                direction=0*np.pi/2,radius=1.,angle=1.,level=1.0), 
        Agent(index='X',atype='mcts_min',position=(8,0),
                direction=1*np.pi/2,radius=1.,angle=1.,level=1.0),  
            ],
    'adhoc_agent_index' : 'A',
    'impostor_index': 'X',
    'tasks' : [
        Task(index='0',position=(3,3),level=0.5),
        Task(index='1',position=(7,7),level=0.5),
        Task(index='2',position=(7,3),level=0.5),
        Task(index='3',position=(3,7),level=0.5),
        Task(index='4',position=(5,5),level=0.5),
            ],
    'obstacles':[]
}

COOP_S9X9_4AGENTS_5TASKS_1ADV = {
    # Scenario 3: FO Adversarial - coop benchmark scenario for adversaries
    # 1 adhoc agent 3 teammate 1 adversary and 5 tasks
    'dim': (9,9),
    'type_knowledge': False,
    'template_types':['l4','l5','l6','adversary'],
    'parameter_knowledge': False,
    'vision_block': True,
    'agents' : [
        Agent(index='A',atype=None,position=(0,0),
                direction=1*np.pi/2,radius=1.,angle=1.,level=0.1), 
        Agent(index='1',atype='l1',position=(0,4),
                direction=0*np.pi/2,radius=1.,angle=1.,level=0.3), 
        Agent(index='2',atype='l2',position=(0,8),
                direction=0*np.pi/2,radius=1.,angle=1.,level=0.4), 
        Agent(index='3',atype='l3',position=(4,0),
                direction=0*np.pi/2,radius=1.,angle=1.,level=0.5), 
        Agent(index='X',atype='mcts_min',position=(8,0),
                direction=1*np.pi/2,radius=1.,angle=1.,level=0.6), 
            ],
    'adhoc_agent_index' : 'A',
    'impostor_index': 'X',
    'tasks' : [
        Task(index='0',position=(3,3),level=0.1),
        Task(index='1',position=(7,7),level=0.2),
        Task(index='2',position=(7,3),level=0.4),
        Task(index='3',position=(3,7),level=0.6),
        Task(index='4',position=(5,5),level=0.8),
            ],
    'obstacles':[]
}