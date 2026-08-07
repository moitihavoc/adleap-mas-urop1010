from src.reasoning.estimation import type_parameter_estimation
from src.reasoning.node import RhoANode, RhoONode
from src.reasoning.node import find_new_rho_root, particle_revigoration

import random
import time

class TBRhoPOMCP(object):

    def __init__(self,max_depth,max_it,kwargs):
        ###
        # Tree Search parameters
        ###
        self.root = None
        self.max_depth = max_depth
        self.max_it = max_it

        self.c                  = kwargs.get("c", 0.5)
        self.discount_factor    = kwargs.get("discount_factor", 0.95)
        self.pr                 = kwargs.get('particle_revigoration',True)
        self.k                  = kwargs.get('k', 100)
        self.time_budget        = kwargs.get('time_budget', 1.0)#float('inf'))
        
        self.smallbag_size      = kwargs.get('smallbag_size', 10)
              
        ###
        # Further settings
        ###
        self.target = kwargs.get('target','max')
        self.initial_target = kwargs.get('target','max')
        self.adversary          = kwargs.get('adversary', False)

        ###
        # Evaluation
        ###
        self.rollout_total_time = 0.0
        self.rollout_count = 0.0
        
        self.simulation_total_time = 0.0
        self.simulation_count = 0.0

    def rhofunction(self, particles, action):
        belief_reward = 0.0
        norm = 0.0
        for p in particles:
            state = p[0]
            weight = p[1]

            tmp_state = state.copy()
            _, reward, _, _ = tmp_state.step(action)

            belief_reward += weight * reward
            norm += weight
        if norm == 0:
            return 0
        return belief_reward / norm
    
    def rhofunction_smallbag(self, smallbag, action):
        belief_reward = 0.0
        for state in smallbag:
            tmp_state = state.copy()
            _, reward, _, _ = tmp_state.step(action)
            belief_reward += reward
        return belief_reward / len(smallbag)

    def importance_sampling(self,smallbag,next_state,action,next_obs):
        next_smallbag = []
        next_smallbag.append(next_state)
        while len(next_smallbag) < self.smallbag_size:
            # (1) sampling the particle from smallbag
            particle = random.choice(smallbag)

            # (2) generating particle' from particle using G
            tmp_state = particle.copy()
            state, _, _, _ = tmp_state.step(action)

            # (3) storing the generated particle particle' in the new smallbag
            if state.observation_is_equal(next_obs):
                next_smallbag.append(state)
            else:
                next_smallbag.append(next_state)
        return next_smallbag
    
    def change_paradigm(self):
        if self.target == 'max':
            return 'min'
        elif self.target == 'min':
            return 'max'
        else:
            raise NotImplemented

    def simulate_action(self, node, action):
        # 1. Copying the current state for simulation
        tmp_state = node.state.copy()

        # 2. Acting
        next_state,reward, _, _ = tmp_state.step(action)
        next_node = RhoANode(action,next_state,node.depth+1,node)

        # 3. Returning the next node and the reward
        return next_node, reward

    def rollout_policy(self,state):
        if getattr(state,'default_policy',None) is not None:
            return state.default_policy()
        return random.choice(state.get_actions_list())

    def rollout(self,node,smallbag):
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
        next_obs = next_state.get_observation()
        node.observation = next_obs
        node.depth += 2

        next_smallbag = self.importance_sampling(smallbag, next_state, action, next_obs)

        end_t = time.time()
        self.rollout_total_time += (end_t - start_t)

        # 4. Rolling out
        return self.rhofunction_smallbag(smallbag, action) +\
            self.discount_factor*self.rollout(node,next_smallbag)

    def get_rollout_node(self,node):
        obs = node.state.get_observation()
        tmp_state = node.state.copy()
        depth = node.depth
        return RhoONode(observation=obs,state=tmp_state,depth=depth,parent=None)

    def is_leaf(self, node):
        if node.depth >= self.max_depth + 1:
            return True
        return False

    def is_terminal(self, node):
        return node.state.state_set.is_final_state(node.state)

    def simulate(self, node, smallbag):
        # 1. Checking the stop condition
        if node.depth == 0:
            node.visits += 1

        if self.is_terminal(node) or self.is_leaf(node):
            return 0

        # 2. Checking child nodes
        if node.children == []:
            # a. adding the children
            for action in node.actions:
                (next_node, _) = self.simulate_action(node, action)
                node.children.append(next_node)
            rollout_node = self.get_rollout_node(node)
            return self.rollout(rollout_node,smallbag)

        self.simulation_count += 1
        start_t = time.time()

        # 3. Selecting the best action
        action = node.select_action(coef=self.c,mode=self.target)

        # 4. Simulating the action
        (action_node, _) = self.simulate_action(node, action) 
        observation = action_node.state.get_observation()

        # 5. Adding the action child on the tree
        if action_node.action in [c.action for c in node.children]:
            for child in node.children:
                if action_node.action == child.action:
                    child.state = action_node.state.copy()
                    action_node = child
                    break
        else:
            node.children.append(action_node)
        action_node.visits += 1

        # 6. Getting the observation and adding the observation child on the tree
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

        end_t = time.time()
        self.simulation_total_time += (end_t - start_t)

        # 7. Generating the new smallbag
        next_smallbag = self.importance_sampling(smallbag,observation_node.state, action, observation)

        # 8. Updating the particle filter
        for state in smallbag:
            node.particle_filter.append(state.copy())
        node.particle_filter.append(node.state.copy())

        # 8. Calculating the reward, quality and updating the node
        for state in next_smallbag:
            weight = state.get_obs_p(action)
            observation_node.add_to_cummulative_bag(state, weight)

        R = self.rhofunction(node.cummulative_bag, action) + \
            float(self.discount_factor * self.simulate(observation_node,next_smallbag))
        node.update(action, R)
        return R

    def search(self, node, agent):
        # 1. Performing the Monte-Carlo Tree Search
        it = 0
        start_t = time.time()
        while (time.time() - start_t < self.time_budget):
            self.target = self.initial_target
            
            # a. Sampling the belief state for simulation
            if len(node.particle_filter) < 1 + self.smallbag_size:
                sampled_states = node.state.sample_nstate(agent, 1 + self.smallbag_size)
                beliefState, smallbag = sampled_states[0], sampled_states[1:]
                for state in sampled_states:
                    node.add_to_cummulative_bag(state, 1.0)
            else:
                sampled_states = random.sample(node.particle_filter, 1 + self.smallbag_size)
                beliefState, smallbag = sampled_states[0], sampled_states[1:]
                for state in sampled_states:
                    node.add_to_cummulative_bag(state, 1.0)
            node.state = beliefState

            # b. simulating
            self.simulate(node,smallbag)
            it += 1

        self.target = self.initial_target
        return node.get_best_action()

    def planning(self, state, agent):
        # 1. Getting the current state and previous action-observation pair
        previous_action = agent.next_action
        current_observation = state.get_observation()

        # 2. Defining the root of our search tree
        # via initialising the tree
        if self.root is None:
            self.root = RhoONode(observation=None,state=state,depth=0,parent=None)
        # or advancing within the existent tree
        else:
            self.root = find_new_rho_root(state, previous_action,\
             current_observation, agent, self.root, adversary=self.adversary)

        # 3. Estimating the parameters 
        if 'estimation_method' in agent.smart_parameters:
            self.root.state, agent.smart_parameters['estimation'] = \
             type_parameter_estimation(self.root.state, agent, agent.smart_parameters\
              ['estimation_method'], **agent.smart_parameters['estimation_args'])

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

def tbrhopomcp_planning(env, agent, max_depth=20, max_it=1000, **kwargs):    
    # 1. Setting the environment for simulation
    copy_env = env.copy()
    copy_env.simulation = True

    # 2. Planning
    tbrhopomcp = TBRhoPOMCP(max_depth, max_it, kwargs) if 'tbrhopomcp' not \
     in agent.smart_parameters else agent.smart_parameters['tbrhopomcp']
     
    # - planning
    next_action, info = tbrhopomcp.planning(copy_env,agent)

    # 3. Updating the search tree
    agent.smart_parameters['tbrhopomcp'] = tbrhopomcp
    agent.smart_parameters['count'] = info
    return next_action,None