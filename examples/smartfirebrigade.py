###
# IMPORTS
###
from multiprocessing import freeze_support
from time import time
import random as rd
import sys
import os
sys.path.append(os.getcwd())

rd.seed(13)
from src.envs.SmartFireBrigadeEnv import SmartFireBrigadeEnv, Drone

###
# Setting the environment
###
def main():
    display = True
    dim = (1000,600)
    max_time = 20

    drones = [
        Drone(index='0',atype='rtmcts',position=[100,100])
    ]
    components = {'drones' : drones,'adhoc_agent_index' : '0'}
    env = SmartFireBrigadeEnv(dim,components,max_time,display=display)
    state = env.reset()

    ###
    # ADLEAP-MAS MAIN ROUTINE
    ###
    done, start_time  = False, time()
    while (time() - start_time) < max_time and not done:
        action = env.listen_action()
        state,reward,done,_ = env.step(action)
        if action['0'] != 0:
            print('Time: %.2f | FPS: %d | Reward: %f |' \
                % (time() - start_time,int(env.episode/(time()-start_time+0.00005)),reward), action)
    env.close()

# WINDOWS SAFE PARALLEL EXECUTION
if __name__ == '__main__':
    freeze_support()
    main()
###
# THE END - That's all folks :)
###
