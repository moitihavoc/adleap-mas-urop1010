from src.envs.InfiltrationEnv import ENTITY_SIZE_SQ
import numpy as np

#####
# Naive target seeking
#####
# returns the vector pointing directly at the target superposed
# with the average vector pointing away from the closest boids *2


def naive_planning(env, agent):
    next_action = (env.components["target"]-agent.position)
    next_action /= np.linalg.norm(next_action)
    for entity in env.components["boids"]:
        if entity.index == env.components["adhoc_agent_index"]:
            continue
        def_to_agent = agent.position-entity.positions
        sq_dists = np.sum((def_to_agent)**2, axis=0)
        mask = sq_dists < ENTITY_SIZE_SQ*2
        if np.any(mask):
            next_action += 2*(np.where(mask, def_to_agent/np.linalg.norm(def_to_agent), np.zeros_like(def_to_agent)).sum(0)/sum(mask))

    return next_action, env.components["target"]