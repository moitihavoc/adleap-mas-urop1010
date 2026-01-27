from src.envs.scenarios.levelforaging.single_agent import \
    THE_CORRIDOR, U_SHAPED, U_OBSTACLES, THE_WAREHOUSE, THE_OFFICE
from src.envs.scenarios.levelforaging.multiagent import \
    MAS_DUO,MAS_STRONG,MAS_WEAK,MAS_MIX,MAS_WRONG,MAS_5TEAM, MAS_7TEAM,\
        MAS_9TEAM, MAS_11TEAM
from src.envs.scenarios.levelforaging.adversarial import \
    S5X5_2AGENTS_1TASKS_1ADV, S9X9_2AGENTS_5TASKS_1ADV, \
        S9X9_4AGENTS_5TASKS_1ADV, COOP_S9X9_4AGENTS_5TASKS_1ADV

MAX_SCENARIO_ID = 16
SINGLE_AGENT_SCENARIOS_ID = [0,1,2,3,4]
MULTI_AGENT_SCENARIOS = [5,6,7,8,9,10,11,12,13]
ADVERSARIAL_SCENARIOS = [14,15,16,17]

def get(method):
    default_scenarios_components = [
         # SINGLE AGENT SCENARIOS
        THE_CORRIDOR, U_SHAPED, U_OBSTACLES, THE_WAREHOUSE, THE_OFFICE,
        # MULTI AGENT SCENARIOS
        MAS_DUO,MAS_STRONG,MAS_WEAK,MAS_MIX,MAS_WRONG,MAS_5TEAM,
        MAS_7TEAM,MAS_9TEAM,MAS_11TEAM,
        # ADVERSARIAL SCENARIOS
        S5X5_2AGENTS_1TASKS_1ADV, S9X9_2AGENTS_5TASKS_1ADV, 
        S9X9_4AGENTS_5TASKS_1ADV, COOP_S9X9_4AGENTS_5TASKS_1ADV 
    ]
    for i in range(len(default_scenarios_components)):
        default_scenarios_components[i]['agents'][0].type = method
    return default_scenarios_components