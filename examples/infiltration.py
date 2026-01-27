###
# IMPORTS
###
from importlib import import_module
from multiprocessing import freeze_support
import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from src.envs.InfiltrationEnv import InfiltrationEnv, Flock, Agent, BoidParams, Difficulties



GRID_WIDTH = 1200
GRID_HEIGHT = 600

#rd.seed(13)

###
# Setting the environment
###
def main():
    display = True
    dim = (GRID_WIDTH,GRID_HEIGHT)
    estimation_method = "naive"
    boid_count = 50
    adhoc_agent_index = 1
    boids=[]
    target = np.array((GRID_WIDTH/3,GRID_HEIGHT/2),dtype="float64")
    waypoints = target + Difficulties.EASY.get_waypoints()
    try:
        module = import_module('src.reasoning.infiltration.'+estimation_method)
    except:
        module = import_module('src.reasoning.'+estimation_method)
    planning_method = getattr(module, estimation_method + '_planning')

    boids.append(Flock.generate_flock(0, "boid", waypoints, BoidParams.get_default(), boid_count))
    boids.append(Agent(1, estimation_method, (5*GRID_WIDTH/6,GRID_WIDTH/2), np.array([0,0])))

    components = {
        'target' : target,
        'boids' : boids,
        'waypoints' : waypoints,
        'adhoc_agent_index' : adhoc_agent_index 
    }

    for b in components['boids']:
        try:
            os.remove('./tmp/'+b.index+'.txt')
        except:
            pass

    env = InfiltrationEnv(dim, components,display=display)
    state = env.reset()

    ###
    # ADLEAP-MAS MAIN ROUTINE
    ###
    done, max_episode = False, 10000000
    adhoc_agent = env.get_adhoc_agent()
    while env.episode < max_episode and not done:
        # 1. Importing agent method
        method = env.import_method(adhoc_agent.type)

        # 2. Reasoning about next action and target
        adhoc_agent.next_action, _ = method(state, adhoc_agent)
        adhoc_agent.next_action = (0,0)

        # 3. Taking a step in the environment
        state,_,done,_ = env.step(action=adhoc_agent.next_action)
        # action, _ = planning_method(env, env.components["boids"][adhoc_agent_index])
        # state,reward,done,info = env.step(action)
        print('Episode:',env.episode)
    env.close()

# WINDOWS SAFE PARALLEL EXECUTION
if __name__ == '__main__':
    freeze_support()
    main()
###
# THE END - That's all folks :)
###
