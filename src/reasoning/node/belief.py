import numpy as np
import random

from src.reasoning.node.pomcp import ANode, ONode

class BeliefANode(ANode):

    def __init__(self,action, state, depth, parent=None):
        super(BeliefANode,self).__init__(action,state,depth,parent)
        self.action = action
        self.observation = None

    def add_child(self,observation):
        state = self.state.copy()
        child = BeliefONode(observation,state,self.depth+1,self)
        self.children.append(child)
        return child

class BeliefONode(ONode):

    def __init__(self,observation, state, depth, parent=None):
        super(BeliefONode,self).__init__(None,state,depth,parent)
        self.action = None
        self.observation = observation
        
        self.particle_filter = []
        self.belief_set = {}
    
    def add_child(self,state,action):
        child = BeliefANode(action,state,self.depth+1,self)
        self.children.append(child)
        return child
    
    def get_child(self,action):
        for c in self.children:
            if c.action == action:
                return c
        return None
    
    def update_belief(self, particle):
        self.particle_filter.append(particle)

        key = particle.hash_state() + particle.hash_observation()
        if key in self.belief_set:
            self.belief_set[key][1] += 1
        else:
            self.belief_set[key] = [particle, 1]


def find_new_belief_root(
 current_state, previous_action, current_observation, agent, previous_root, adversary=False
) -> BeliefONode:
    # 1. If the root doesn't exist yet, create it
    # - NOTE: The root is always represented as an "observation node" since the 
    # next node must be an action node.
    if previous_root is None:
        new_root = BeliefONode(observation=None,state=current_state,depth=0,parent=None)
        #print('<!> Creating new root node: no previous root found')
        return new_root

    # 2. Else, walk on the tree to find the new one (giving the previous information)
    action_node, observation_node, new_root = None, None, None

    # a. walking over action nodes
    for child in previous_root.children:
        if child.action == previous_action:
            action_node = child
            break

    # - if we didn't find the action node, create a new root
    if action_node is None:
        new_root = BeliefONode(observation=None,state=current_state,depth=0,parent=None)
        #print('<!> Creating new root node: no action node found')
        #print('<!> Previous action:', previous_action)
        return new_root

    # b. walking over observation nodes
    for child in action_node.children:
        if child.state.observation_is_equal(current_observation):
            observation_node = child
            break

    # - if we didn't find the action node, create a new root
    if observation_node is None:
        new_root = BeliefONode(observation=None,state=current_state,depth=0,parent=None)
        #print('<!> Creating new root node: no observation node found')
        return new_root
    
    # c. checking if we are in an adversarial setting
    if adversary:
        action_node, observation_node = None, None
        for child in new_root.children:
            if child.action == agent.smart_parameters['adversary_last_action']:
                action_node = child
                break
        # - if we didn't find the action node, create a new root'
        if action_node is None:
            new_root = BeliefONode(\
                observation=None,state=current_state,depth=0,parent=None)
            return new_root

        for child in action_node.children:
            if child.state.observation_is_equal(\
             agent.smart_parameters['adversary_last_observation']):
                observation_node = child
                break
        # - if we didn't find the action node, create a new root
        if observation_node is None:
            new_root = BeliefONode(\
                observation=None,state=current_state,depth=0,parent=None)
            return new_root

    # 3. Definig the new root and updating the depth
    new_root = observation_node
    new_root.parent = None
    new_root.update_depth(0)
    #print('<y> Walking on the tree to find the new root node')
    return new_root