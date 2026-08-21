import math
import numpy as np
from src.envs import LevelForagingEnv
from src.envs.LevelForagingEnv import is_visible

def find_visible_grids(env: LevelForagingEnv):
    """
    Find and return all visible grids given the ad hoc agent.
    """
    dim = env.shape
    agent = env.get_adhoc_agent()
    visible_grids = []

    # find the agent's view radius and angle of view
    max_distance = math.sqrt(dim[0]**2 + dim[1]**2)
    view_radius = agent.radius * max_distance
    view_angle = agent.angle * 2 * np.pi
    obstacles = env.components['obstacles']
    vision_block = env.vision_block

    for x in range(dim[0]):
        for y in range(dim[1]):
            grid_pos = (x, y)
            if is_visible(
                obj = grid_pos,
                viewer = agent.position,
                direction = agent.direction,
                radius = view_radius,
                angle = view_angle,
                obstacles = obstacles,
                vision_block = vision_block

            ):
                visible_grids.append(grid_pos)
    return visible_grids