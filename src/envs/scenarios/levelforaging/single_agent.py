import numpy as np
from src.envs.LevelForagingEnv import Agent, Task

THE_CORRIDOR = {
    # Scenario 0: The Corridor (Rectangle PO Foraging Scenario)
    'dim': (20,2),
    'type_knowledge': True,
    'parameter_knowledge': True,
    'vision_block': True,
    'agents' : [
        Agent(index='A',atype=None,position=(1,1),
                direction=1*np.pi/2,radius=0.2,angle=0.3,level=1.0), 
            ],
    'adhoc_agent_index' : 'A',
    'impostor_index': None,
    'tasks' : [
        Task(index='0',position=(0,0),level=1.0),
        Task(index='4',position=(19,1),level=1.0)
            ],
    'obstacles':[]
}

U_SHAPED = {
    # Scenario 1: The U-shaped Scenario
    'dim': (15, 15),
    'type_knowledge': True,
    'parameter_knowledge': True,
    'vision_block': True,
    'agents': [
    Agent(index='0',atype=None,position=(14,14),direction=1*np.pi/2,radius=0.2,angle=0.3,level=1.0),
    ],
    'adhoc_agent_index': '0',
    'impostor_index': None,
    'tasks': [
            Task(index='0',position=(0,7),level=1.0),
            Task(index='1',position=(14,0),level=1.0),
            Task(index='2',position=(14,12),level=1.0),
    ],
    'obstacles': [
    (3, 3) , (3, 4) , (3, 5) , (3, 6) , (3, 7) , (3, 8) , (3, 9) , (3, 10) , (3, 11) , (4, 3) , (4, 4) , (4, 5) , (4, 6) , (4, 7) , (4, 8) , (4, 9) , (4, 10) , (4, 11) , (5, 3) , (5, 4) , (5, 5) , (5, 6) , (5, 7) , (5, 8) , (5, 9) , (5, 10) , (5, 11) , (6, 3) , (6, 4) , (6, 5) , (6, 6) , (6, 7) , (6, 8) , (6, 9) , (6, 10) , (6, 11), (7, 3) , (7, 4) , (7, 5) , (7, 6) , (7, 7) , (7, 8) , (7, 9) , (7, 10) , (7, 11) , (8, 3) , (8, 4) , (8, 5), (8, 6) , (8, 7) , (8, 8) , (8, 9) , (8, 10) , (8, 11) , (9, 3) , (9, 4) , (9, 5) , (9, 6) , (9, 7) , (9, 8), (9, 9) , (9, 10) , (9, 11) , (10, 3) , (10, 4) , (10, 5) , (10, 6) , (10, 7) , (10, 8) , (10, 9) , (10, 10) , (10, 11) , (11, 3) , (11, 4) , (11, 5) , (11, 6) , (11, 7) , (11, 8) , (11, 9) , (11, 10) , (11, 11) , (12, 3) , (12, 4) , (12, 5) , (12, 6) , (12, 7) , (12, 8) , (12, 9) , (12, 10) , (12, 11) , (13, 3) , (13, 4) , (13, 5) , (13, 6) , (13, 7) , (13, 8) , (13, 9) , (13, 10) , (13, 11) , (14, 3) , (14, 4) , (14, 5) , (14, 6) , (14, 7) , (14, 8) , (14, 9) , (14, 10) , (14, 11) ,
        ]
}

U_OBSTACLES = {
    # Scenario 2: The U-Obstacles Scenario
    'dim': (20, 10),
    'type_knowledge': True,
    'parameter_knowledge': True,
    'vision_block': True,
    'agents': [
        Agent(index='0',atype=None,position=(0,5),direction=0,radius=0.2,angle=0.3,level=1.0),
    ],
    'adhoc_agent_index': '0',
    'impostor_index': None,
    'tasks': [
        Task(index='0',position=(4,5),level=1.0),
        Task(index='1',position=(8,4),level=1.0),
        Task(index='2',position=(12,5),level=1.0),
        Task(index='3',position=(19,0),level=1.0),
        Task(index='4',position=(19,9),level=1.0),
    ],
    'obstacles': [
        (3, 2) , (3, 7) , (4, 2) , (4, 7) , (5, 2) , (5, 3) , (5, 4) , (5, 5) , (5, 6) , (5, 7) , (11, 2) , (11, 3) , (11, 4) , (11, 5) , (11, 6) , (11, 7) , (12, 2) , (12, 7) , (13, 2) , (13, 7) ,
    ],
}

THE_WAREHOUSE = {
    # Scenario 3: The Warehouse (Square PO Foraging Scenario)
    'dim': (20,20),
    'type_knowledge': True,
    'parameter_knowledge': True,
    'vision_block': True,
    'agents' : [
        Agent(index='A',atype=None,position=(10,1),
                direction=1*np.pi/2,radius=0.2,angle=0.3,level=1.0), 
            ],
    'adhoc_agent_index' : 'A',
    'impostor_index': None,
    'tasks' : [
        Task(index='0',position=(11,0),level=1.0),
        Task(index='1',position=(10,18),level=1.0),
        Task(index='2',position=(9,10),level=1.0),
        Task(index='A0',position=(1,1),level=1.0),
            Task(index='A1',position=(3,3),level=1.0),
            Task(index='A2',position=(1,3),level=1.0),
            Task(index='A3',position=(3,1),level=1.0),
        Task(index='B0',position=(18,1),level=1.0),
            Task(index='B1',position=(18,3),level=1.0),
            Task(index='B2',position=(16,3),level=1.0),
            Task(index='B3',position=(16,1),level=1.0),
        Task(index='C0',position=(18,18),level=1.0),
            Task(index='C1',position=(18,16),level=1.0),
            Task(index='C2',position=(16,16),level=1.0),
            Task(index='C3',position=(16,18),level=1.0),
        Task(index='D0',position=(1,18),level=1.0),
            Task(index='D1',position=(3,18),level=1.0),
            Task(index='D2',position=(3,16),level=1.0),
            Task(index='D3',position=(1,16),level=1.0),
            ],
    'obstacles':[]
}

THE_OFFICE = {
    # Scenario 4: The Office (Square with Obstacles PO Foraging Scenario)
    'dim': (15, 10),
    'type_knowledge': True,
    'parameter_knowledge': True,
    'vision_block': True,
    'agents': [
                Agent(index='0',atype=None,position=(0,9),direction=3*np.pi/2,radius=0.2,angle=0.3,level=1.0),
    ],
    'adhoc_agent_index': '0',
    'impostor_index': None,
    'tasks': [
                Task(index='0',position=(0,0),level=1.0),
                Task(index='1',position=(3,3),level=1.0),
                Task(index='2',position=(5,7),level=1.0),
                Task(index='3',position=(11,5),level=1.0),
                Task(index='4',position=(14,0),level=1.0),
    ],
    'obstacles': [
            (2, 2) , (2, 3) , (2, 4) , (2, 5) , (2, 6) , (2, 7) , (2, 8) , (2, 9) , (3, 2) , (3, 6) , (4, 2) , (4, 6) , (5, 2) , (5, 6) , (6, 2) , (6, 3) , (6, 4) , (6, 6) , (6, 7) , (6, 8) , (9, 2) , (9, 3) , (9, 4) , (9, 5) , (9, 6) , (9, 7) , (9, 8) , (10, 2) , (10, 8) , (11, 2) , (11, 4) , (11, 6) , (11, 8) , (12, 2) , (12, 4) , (12, 5) , (12, 6) , (12, 8) , (13, 2) , (13, 8) ,
    ],
}
