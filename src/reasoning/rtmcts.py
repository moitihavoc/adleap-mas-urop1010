from src.reasoning.node import QNode
import random
import time
from src.reasoning.estimation import type_parameter_estimation

class RealTimeMCTS(object):

    def __init__(self,max_depth,max_it,kwargs):
        ###
        # Traditional Monte-Carlo Tree Search parameters
        ###
        self.max_depth = max_depth
        self.max_it = max_it
        self.c = 0.5
        discount_factor = kwargs.get('discount_factor')
        self.discount_factor = discount_factor\
            if discount_factor is not None else 0.95

        ###
        # Further settings
        ###
        target = kwargs.get('target')
        if target is not None:
            self.target = target
            self.initial_target = target
        else: #default
            self.target = 'max'
            self.initial_target = 'max'

        adversary_mode = kwargs.get('adversary')
        if adversary_mode is not None:
            self.adversary = adversary_mode
        else: #default
            self.adversary = False

        stack_size = kwargs.get('state_stack_size')
        if stack_size is not None:
            self.state_stack_size = stack_size
        else: #default
            self.state_stack_size = 30

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

    def simulate_action(self, node, action):
        # 1. Copying the current state for simulation
        tmp_state = node.state.copy()

        # 2. Acting
        next_state,reward, _, _ = tmp_state.step(action)
        next_node = QNode(action,next_state,node.depth+1,node)

        # 3. Returning the next node and the reward
        return next_node, reward

    def rollout_policy(self,state):
        return random.choice(state.get_actions_list())

    def rollout(self,node):
        # 1. Checking if it is an end state or leaf node
        if self.is_terminal(node) or self.is_leaf(node):
            return 0

        self.rollout_count += 1
        start_t = time.time()

        # 2. Choosing an action
        action = self.rollout_policy(node.state)

        # 3. Simulating the action
        next_state, reward, _, _ = node.state.step(action)
        node.state = next_state
        node.depth += 1

        end_t = time.time()
        self.rollout_total_time += (end_t - start_t)

        # 4. Rolling out
        return reward +\
            self.discount_factor*self.rollout(node)

    def get_rollout_node(self,node):
        tmp_state = node.state.copy()
        depth = node.depth
        return QNode(action=None,state=tmp_state,depth=depth,parent=None)

    def is_leaf(self, node):
        if node.depth >= self.max_depth + 1:
            return True
        return False

    def is_terminal(self, node):
        return node.state.state_set.is_final_state(node.state)

    def simulate(self, node):
        # 1. Checking the stop condition
        if self.is_terminal(node) or self.is_leaf(node):
            return 0

        # 2. Checking child nodes
        if node.children == []:
            # a. adding the children
            for action in node.actions:
                (next_node, reward) = self.simulate_action(node, action)
                node.children.append(next_node)
            rollout_node = self.get_rollout_node(node)
            return self.rollout(rollout_node)

        self.simulation_count += 1
        start_t = time.time()
        
        # 3. Selecting the best action
        action = node.select_action(coef=self.c,mode=self.target)
        self.target = self.change_paradigm() if self.adversary else self.target     

        # 4. Simulating the action
        (next_node, reward) = self.simulate_action(node, action)

        # 5. Adding the action child on the tree
        if next_node.action in [c.action for c in node.children]:
            for child in node.children:
                if next_node.action == child.action:
                    child.state = next_node.state.copy()
                    next_node = child
                    break
        else:
            node.children.append(next_node)
        next_node.visits += 1

        end_t = time.time()
        self.simulation_total_time += (end_t - start_t)

        # 7. Calculating the reward, quality and updating the node
        R = reward + float(self.discount_factor * self.simulate(next_node))
        node.update(action, R)
        next_node.visits += 1
        return R

    def search(self, node):
        # Performing the Monte-Carlo Tree Search
        it = 0
        while it < self.max_it:
            self.target = self.initial_target
            self.simulate(node)
            it += 1
        self.target = self.initial_target
        #print('BEST ACTION:',self.target)
        return node.get_best_action(self.target)

    def find_new_root(self,current_state,previous_action, previous_root, adversary_last_action=None):
        # 1. If the root doesn't exist yet, create it
        # - NOTE: The root is always represented as an "observation node" since the next node
        # must be an action node.
        if previous_root is None:
            new_root = QNode(action=None,state=current_state,depth=0,parent=None)
            return new_root

        # 2. Else, walk on the tree to find the new one (giving the previous information)
        new_root = None

        # a. walking over action nodes
        for child in previous_root.children:
            if child.action == previous_action:
                new_root = child
                break

        # - if we didn't find the action node, create a new root
        if new_root is None:
            new_root = QNode(action=None,state=current_state,depth=0,parent=None)
            return new_root

        # b. checking the adversary condition
        if self.adversary:
            for child in new_root.children:
                if child.action == adversary_last_action:
                    new_root = child
                    break

            # - if we didn't find the action node, create a new root
            if new_root is None:
                new_root = QNode(observation=None,state=current_state,depth=0,parent=None)
                return new_root

        # 3. Definig the new root and updating the depth
        new_root.parent = None
        new_root.update_depth(0)
        return new_root

    def planning(self, state, agent):
        # 1. Getting the current state and previous action-observation pair
        previous_action = agent.next_action

        # 2. Defining the root of our search tree
        # via initialising the tree
        if 'search_tree' not in agent.smart_parameters:
            root_node = QNode(action=None,state=state,depth=0,parent=None)
        # or advancing within the existent tree
        else:
            if self.adversary:
                root_node = self.find_new_root(state, previous_action,agent.smart_parameters['search_tree'],\
                    adversary_last_action=agent.smart_parameters['adversary_last_action'])
            else:
                root_node = self.find_new_root(state, previous_action, agent.smart_parameters['search_tree'])
            # if no valid node was found, reset the tree
            if root_node is None:
                root_node = QNode(action=None,state=state,depth=0,parent=None)
        
        # 3. Estimating the parameters 
        if 'estimation_method' in agent.smart_parameters:
            root_node.state, agent.smart_parameters['estimation'] = type_parameter_estimation(root_node.state,agent,\
                agent.smart_parameters['estimation_method'], *agent.smart_parameters['estimation_args'])

        # 4. Searching for the best action within the tree
        best_action = self.search(root_node)

        # 5. Returning the best action
        # root_node.show_qtable()
        
        return best_action, root_node, {'nrollouts': self.rollout_count,'nsimulations':self.simulation_count}

def rtmcts_planning(env, agent, max_depth=20, max_it=100, **kwargs):    
    # 1. Setting the environment for simulation
    copy_env = env.copy()
    copy_env.viewer = None
    copy_env.simulation = True

    # 2. Planning
    mcts = RealTimeMCTS(max_depth, max_it, kwargs)
    next_action, search_tree, info = mcts.planning(copy_env,agent)

    # 3. Updating the search tree
    agent.smart_parameters['search_tree'] = search_tree
    agent.smart_parameters['count'] = info
    return next_action,None