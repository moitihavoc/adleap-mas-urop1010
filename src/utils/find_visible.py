import math
import numpy as np
from src.envs.LevelForagingEnv import is_visible

def find_visible_grids(agent, vb:bool):
    """
    Find and return all visible grids given the ad hoc agent.
    """
    dim = agent.env.shape
    visible_grids = []

    # find the agent's view radius and angle of view
    max_distance = math.sqrt(dim[0]**2 + dim[1]**2)
    view_radius = agent.radius * max_distance
    view_angle = agent.angle * 2 * np.pi
    obstacles = agent.env.components['obstacles']
    vision_block = vb

    for x in range(dim[0]):
        for y in range(dim[1]):
            grid_pos = (x, y)
            if is_visible(
                obj = grid_pos,
                agent_pos = agent.position,
                agent_dir = agent.direction,
                agent_radius = view_radius,
                agent_angle = view_angle,
                obstacles = obstacles,
                vision_block = vision_block

            ):
                visible_grids.append(grid_pos)
    return visible_grids