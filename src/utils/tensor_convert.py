import numpy as np
import torch

def dict_to_tensor(obs_dict:dict, fields:np.ndarray, dim:int, visible_grids:list):
    """
    Combines the original LBF env's observation dictionary with the trace field into a tensor representation for CNN training.
    The tensor has 9 channels:
    - channel 1: agent positions (presence of agents in the grid)
    - channel 2: task positions (presence of tasks in the grid)
    - channel 3: obstacle positions (if any)
    - channel 4: trace intensity
    - channel 5: trace age
    - channel 6: trace agent-level
    - channel 7: trace help signal
    - channel 8: trace claim signal
    - channel 9: visibility mask (1 if visible, 0 otherwise)
    """

    tensor = np.zeros((9, dim, dim), dtype=np.float32)

    for agent in obs_dict['agents']:
        x, y = agent[1],agent[2]
        tensor[0, x, y] = 1.0  

    for task in obs_dict['tasks']:
        x, y = (task[1], task[2]) 
        tensor[1, x, y] = 1.0

    if (obs_dict['obstacles']):
        for obstacle in obs_dict['obstacles']:
            x, y = obstacle.position
            tensor[2, x, y] = 1.0

    # Fill in the trace field channels
    tensor[3:8, :, :] = fields

    # apply visibility mask 
    for (x, y) in visible_grids:
        tensor[8, x, y] = 1.0

    return torch.from_numpy(tensor).float()
