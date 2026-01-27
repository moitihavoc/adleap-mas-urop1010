# Implementation of POMCP based estimation
import numpy as np
import random as rd

class POMCE(object):

    def __init__(self,env,template_types,parameter_minmax,discount_factor=0.9,max_iter=100,max_depth=10,min_particles=100):
        # Initialising the POMCE parameters
        self.template_types = template_types
        self.nparameters = len(parameter_minmax)
        self.parameters_minmax = parameter_minmax
        self.discount_factor = discount_factor
        self.max_iter = max_iter
        self.max_depth = max_depth

        # Minimum number of particles in black-box
        self.min_particles = min_particles
        self.teammate = {}
        self.check_teammates_estimation_set(env)

    def check_teammates_estimation_set(self,env):
        # Initialising the bag for the agents, if it is missing
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            # for each teammate
            tindex = teammate.index
            if tindex != adhoc_agent.index and tindex not in self.teammate:
                self.teammate[tindex] = {}
                
                # for each type
                for type in self.template_types:
                    # create the estimation set and the bag of estimators
                    self.teammate[tindex][type] = {}
                    if env.type_knowledge:
                        self.teammate[tindex][type]['probability_history'] = 1 if type == self.teammate.type else 0
                    else:
                        self.teammate[tindex][type]['probability_history'] = [1/len(self.template_types)]

                    if env.parameter_knowledge:
                        self.teammate[tindex][type]['parameter_estimation_history'] = \
                            [teammate.get_parameters()]
                    else:
                        sampled_param = [rd.uniform(self.parameters_minmax[n][0],\
                            self.parameters_minmax[n][1]) for n in range(self.nparameters)]
                        
                        self.teammate[tindex][type]['parameter_estimation_history'] = \
                            [sampled_param]

    def get_type_info(self, root_node, agent):
        # to occount of PO : total occurence != len(root.particles)
        total_occurence = 0

        # building the type dict
        type_dict = {}
        for t in self.template_types:
            type_dict[t] = 0

        # counting the occurance of each type
        for sampled_env in root_node.particle_filter:
            for ag in sampled_env.components['agents']:
                if ag.index == agent.index:
                    type_dict[ag.type]+=1
                    total_occurence += 1
                    break

        return type_dict, total_occurence
    
    def update(self,env,adhoc_agent):
        # 1. Checking if there is a search tree to estimate the
        # types and parameters
        pomcp_based = [key for key in adhoc_agent.smart_parameters.keys() if 'pomcp' in key.lower()]
        tree_based = [key for key in adhoc_agent.smart_parameters.keys() if 'tree' in key.lower()]

        if pomcp_based:
            if adhoc_agent.smart_parameters[pomcp_based[0]].root is None:
                #print('= No search tree')
                return self
            else:
                root_node = adhoc_agent.smart_parameters[pomcp_based[0]].root 
        elif tree_based:
            if adhoc_agent.smart_parameters[tree_based[0]].root is None:
                #print('= No search tree')
                return self
            else:
                root_node = adhoc_agent.smart_parameters[tree_based[0]].root 
        else:
            return self

        # 2. Checking if the estimation sets are created
        self.check_teammates_estimation_set(env)

        # 3. Performing the estimation
        for agent in env.components['agents']:
            # if the agent is the adhoc agent, skip
            if (agent.index == adhoc_agent.index):
                continue
    
            # extracting type information
            type_dict, total_occurence = self.get_type_info(root_node, agent)
            if sum(type_dict.values()) == 0:
                for t in type_dict:
                    type_dict[t] += 1
                    total_occurence += 1

            # extracting parameters information
            for t in self.template_types:
                # updating the estimation history
                if(type_dict[t]==0):
                    prev_est = self.teammate[agent.index][t]['parameter_estimation_history'][-1]
                    self.teammate[agent.index][t]['parameter_estimation_history'].append(prev_est)
                    self.teammate[agent.index][t]['probability_history'].append(0)
                    continue
                parameter = np.array([0 for i in range(0,self.nparameters)],dtype="float")
                
                # calculating the total parameter value
                if len(root_node.particle_filter) != 0:
                    for sampled_env in root_node.particle_filter:
                        for ag in sampled_env.components["agents"]:
                            if agent.index ==ag.index:
                                sampled_ag = ag
                                break
                        
                        if (sampled_ag.index == ag.index) and (sampled_ag.type == t):
                            parameter += sampled_ag.get_parameters()
                    
                    # calculating the mean parameters for the type t
                    parameter = parameter/type_dict[t]
                else:
                    prev_est = self.teammate[agent.index][t]['parameter_estimation_history'][-1]
                    parameter = prev_est
                    
                # adding to the estimation set
                self.teammate[agent.index][t]['parameter_estimation_history'].append(parameter)
                self.teammate[agent.index][t]["probability_history"].append(type_dict[t]/total_occurence)
        return self

    def sample_type_for_agent(self, teammate):
        type_prob = np.zeros(len(self.template_types))
        for i in range(len(self.template_types)):
            type = self.template_types[i]
            type_prob[i] = self.teammate[teammate.index][type]['probability_history'][-1]
        
        sampled_type = rd.choices(self.template_types,type_prob,k=1)
        return sampled_type[0]

    def get_parameter_for_selected_type(self, teammate, selected_type):
        parameter_est = self.teammate[teammate.index][selected_type]['parameter_estimation_history'][-1]
        return parameter_est            
    
    def sample_parameter_for_selected_type(self, teammate, selected_type):
        parameter_est = []
        for i in range(len(self.parameters_minmax)):
            min, max = self.parameters_minmax[i][0],self.parameters_minmax[i][1]
            parameter = self.teammate[teammate.index][selected_type]['parameter_estimation_history'][-1][i]
            
            noise = (1 - self.teammate[teammate.index][selected_type]['probability_history'][-1])*(np.random.uniform(min,max))
            parameter += noise
            parameter_est.append(np.clip(parameter,min,max))
        return parameter_est       

    def sample_state(self, env):
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            if teammate.index != adhoc_agent.index:
                selected_type = self.sample_type_for_agent(teammate) \
                    if env.type_knowledge is False else teammate.type
                selected_parameter = self.sample_parameter_for_selected_type(teammate,selected_type)\
                    if env.parameter_knowledge is False else teammate.get_parameters()

                teammate.type = selected_type
                teammate.set_parameters(selected_parameter)

        return env

    def get_estimation(self,env):
        type_probabilities, estimated_parameters, indexes = [], [], []
    
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            if teammate.index != adhoc_agent.index:
                indexes.append(teammate.index)   
                if teammate.index not in self.teammate.keys():
                    # type results
                    type_prob = np.array([-1 for i in range(0,len(self.template_types))]) 
                    # parameter result
                    parameter_est = [np.array([-1 for i in range(0,self.nparameters)]) for type in self.template_types]

                else:
                    # type result
                    type_prob = [\
                        self.teammate[teammate.index][type]['probability_history'][-1] \
                            for type in self.template_types ]
                    # parameter result
                    parameter_est = [\
                        self.teammate[teammate.index][type]['parameter_estimation_history'][-1] \
                            for type in self.template_types]
            
                type_probabilities.append(list(type_prob))
                estimated_parameters.append(list(parameter_est))

        return type_probabilities, estimated_parameters, indexes

    def show_estimation(self, env, mode='both'):
        type_probabilities, estimated_parameters, indexes = self.get_estimation(env)
        types = [t for t in self.template_types]

        if mode == 'both' or mode == 'type':
            print('|%10s|' %('Type'),end='')
            for type in types:
                print('|%10s|' %(str(type)),end='')
            print('')

            for i in range(len(type_probabilities)):
                print('|%10s|' %('Agent '+str(indexes[i])), end='')
                for j in range(len(type_probabilities[i])):
                    print('|%.8f|' %(type_probabilities[i][j]), end='')
                print('')
        
        if  mode == 'both' or mode == 'parameter':
            print('|%10s| ==========' %('Parameters'))
            for i in range(len(estimated_parameters)):
                print('|xxxxxxxxxx| Agent %3s:' %(indexes[i]))
                for j in range(len(types)-1):
                    print('|xxxxxxxxxx|xxxxx| Type %3s:' %(types[j]), estimated_parameters[i][j])


