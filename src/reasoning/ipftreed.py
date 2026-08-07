from src.reasoning.node import ANode, ONode
from src.reasoning.node import find_new_PO_root, particle_revigoration

import numpy as np
import random
import time
from src.reasoning.estimation import type_parameter_estimation

class IPFTreeD(object):

    def __init__(self,max_depth,max_it,kwargs):
        ###
        # Traditional Monte-Carlo Tree Search parameters
        ###
        self.root = None
        self.max_depth = max_depth
        self.max_it = max_it

        self.c                  = kwargs.get("c", 0.5)
        self.discount_factor    = kwargs.get("discount_factor", 0.95)
        self.pr                 = kwargs.get('particle_revigoration',True)
        self.k                  = kwargs.get('k', 100)
        self.m                  = kwargs.get('m', 20) # small particle filter size
        self.eta                = kwargs.get('eta', 1/60) # information gain weight
        self.time_budget        = kwargs.get('time_budget', 10.0)#float('inf'))

        ###
        # Further settings
        ###
        self.target             = kwargs.get('target','max')
        self.initial_target     = kwargs.get('target','max')
        self.adversary          = kwargs.get('adversary', False)

        ###
        # Evaluation
        ###
        self.rollout_total_time = 0.0
        self.rollout_count = 0.0
        
        self.simulation_total_time = 0.0
        self.simulation_count = 0.0

    def change_paradigm(self):
        if self.target == 'max':
            return 'min'
        elif self.target == 'min':
            return 'max'
        else:
            raise NotImplemented
        
    def calculate_belief_entropy(self,belief):
        belief_weight = self.get_particle_weights(belief)
        entropy = 0.0
        for i in range(len(belief)):
            entropy += belief_weight[i]*np.log(belief_weight[i])
        return -entropy

    def rhofunction(self,belief,next_belief,task_reward):
        belief_entropy = self.calculate_belief_entropy(belief)
        next_belief_entropy = self.calculate_belief_entropy(next_belief)
        information_gain = next_belief_entropy - belief_entropy
        return task_reward + self.eta * information_gain

    def get_particle_weights(self, belief):
        weights = []
        obs = [p.get_observation() for p in belief]
        obs_dict = {}
        for o in obs:
            k = str(o)
            if k not in obs_dict:
                obs_dict[k] = 1
            else:
                obs_dict[k] += 1

        for o in obs:
            k = str(o)
            weights.append(obs_dict[k]/len(belief))
        return weights
        
    def simulate_belief(self, belief, action, next_node):
        next_belief = []
        task_reward = 0.0
        observation = next_node.observation

        weights = self.get_particle_weights(belief)
        for i in range(len(belief)):
            tmp_state = belief[i].copy()
            next_state, reward, _, _ = tmp_state.step(action)
            if next_state.observation_is_equal(observation):
                task_reward += weights[i]*reward  
                next_belief.append(next_state)
            else:
                next_belief.append(next_node.state.copy())
        return next_belief, task_reward

    def simulate_action(self, node, action):
        # 1. Copying the current state for simulation
        tmp_state = node.state.copy()

        # 2. Acting
        next_state,reward, _, _ = tmp_state.step(action)
        next_node = ANode(action,next_state,node.depth+1,node)

        # 3. Returning the next node and the reward
        return next_node, reward

    def rollout_policy(self,state):
        if getattr(state,'default_policy',None) is not None:
            return state.default_policy()
        return random.choice(state.get_actions_list())

    def rollout(self,node,belief):
        # 1. Checking if it is an end state or leaf node
        if self.is_terminal(node) or self.is_leaf(node):
            return 0

        self.rollout_count += 1
        start_t = time.time()

        # 2. Choosing an action
        action = self.rollout_policy(node.state)

        # 3. Simulating the action
        next_state, _, _, _ = node.state.step(action)
        node.state = next_state
        observation = next_state.get_observation()
        node.observation = observation
        node.depth += 2

        next_belief, task_reward = self.simulate_belief(belief, action, node)
        reward = self.rhofunction(belief,next_belief,task_reward)

        end_t = time.time()
        self.rollout_total_time += (end_t - start_t)

        # 4. Rolling out
        return reward +\
            self.discount_factor*self.rollout(node, next_belief)

    def get_rollout_node(self,node):
        obs = node.state.get_observation()
        tmp_state = node.state.copy()
        depth = node.depth
        return ONode(observation=obs,state=tmp_state,depth=depth,parent=None)

    def is_leaf(self, node):
        if node.depth >= self.max_depth + 1:
            return True
        return False

    def is_terminal(self, node):
        return node.state.state_set.is_final_state(node.state)

    def simulate(self, node, belief):
        # 1. Checking the stop condition
        if node.depth == 0:
            node.visits += 1

        if self.is_terminal(node) or self.is_leaf(node):
            return 0

        self.simulation_count += 1
        start_t = time.time()

        # 2. Expanding action nodes
        if node.children == []:
            # a. adding the children
            for action in node.actions:
                (next_node, _) = self.simulate_action(node, action)
                node.children.append(next_node)
            rollout_node = self.get_rollout_node(node)
            return self.rollout(rollout_node, belief)
        
        # 3. Selecting an action and getting the action node
        action = node.select_action(coef=self.c,mode=self.target)
        
        (action_node, _) = self.simulate_action(node, action)
        observation = action_node.state.get_observation()

        if action_node.action in [c.action for c in node.children]:
            for child in node.children:
                if action_node.action == child.action:
                    child.state = action_node.state.copy()
                    action_node = child
                    break
        else:
            node.children.append(action_node)
        action_node.visits += 1

        # 4. Getting observation node
        observation_node = None
        for child in action_node.children:
            if child.state.observation_is_equal(observation):
                observation_node = child
                observation_node.state = action_node.state.copy()
                observation_node.particle_filter.append(action_node.state)
                break
        
        if observation_node is None:
            observation_node = action_node.add_child(observation)
            observation_node.particle_filter.append(observation_node.state)
        observation_node.visits += 1

        # 4. Simulating the action through the current belief
        next_belief, task_reward = self.simulate_belief(belief, action, observation_node)
        reward = self.rhofunction(belief,next_belief,task_reward)

        end_t = time.time()
        self.simulation_total_time += (end_t - start_t)

        # . Calculating the reward, quality and updating the node
        R = reward + float(self.discount_factor * self.simulate(observation_node, next_belief))
        node.particle_filter.append(node.state)
        node.update(action, R)
        return R

    def search(self, node, agent):
        # 1. Performing the Monte-Carlo Tree Search
        it = 0
        start_t = time.time()
        while (time.time() - start_t < self.time_budget):
            self.target = self.initial_target

            # a. Sampling the belief state for simulation
            if len(node.particle_filter) < self.k:
                beliefState = node.state.sample_state(agent)
                small_particle_set = node.state.sample_nstate(agent, self.m)
            else:
                beliefState = random.sample(node.particle_filter,1)[0]
                small_particle_set = random.choices(node.particle_filter,k=self.m)
            node.state = beliefState

            # b. simulating
            self.simulate(node,small_particle_set)
            it += 1

        self.target = self.initial_target
        return node.get_best_action(self.target)

    def planning(self, state, agent):
        # 1. Getting the current state and previous action-observation pair
        previous_action = agent.next_action
        current_observation = state.get_observation()

        # 2. Defining the root of our search tree
        # via initialising the tree
        if self.root is None:
            self.root = ONode(observation=None,state=state,depth=0,parent=None)
        # or advancing within the existent tree
        else:
            self.root = find_new_PO_root(state, previous_action,\
             current_observation, agent, self.root, adversary=self.adversary)

        # 4. Performing particle revigoration
        #if self.pr:
        #    particle_revigoration(state,agent,self.root,self.k)

        # 5. Searching for the best action within the tree
        best_action = self.search(self.root, agent)

        # 6. Returning the best action
        self.root.show_qtable()
        info = { 'nrollouts': self.rollout_count,
            'nsimulations':self.simulation_count}
        return best_action, info

def ipftreed_planning(env, agent, max_depth=20, max_it=1000, **kwargs):    
    # 1. Setting the environment for simulation
    copy_env = env.copy()
    copy_env.simulation = True

    # 2. POMCP Planning
    # - initialising/getting the plannin algorithm
    ipftree_d = IPFTreeD(max_depth, max_it, kwargs) if 'ipftree_d' not \
     in agent.smart_parameters else agent.smart_parameters['ipftree_d']
    
    # - planning
    next_action, info = ipftree_d.planning(copy_env,agent)

    # 3. Updating the search tree
    agent.smart_parameters['ipftree_d'] = ipftree_d
    agent.smart_parameters['count'] = info
    return next_action,None