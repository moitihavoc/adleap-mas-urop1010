import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from time import time
from typing import Type

from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import SGD
import tensorflow as tf
import keras

def model_exists(env_name,target_agent):
    file_name = './src/reasoning/trainedmodels/'+\
     env_name+'_'+target_agent+'_model.h5'
    if os.path.exists(file_name):
        return True
    return False

def save_model(model,env_name,target_agent):
    file_name = './src/reasoning/trainedmodels/'+\
     env_name+'_'+target_agent+'_model'
    # serialize model to JSON
    model_json = model.to_json()
    with open(file_name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(file_name+".h5")
        
def load_model(env_name,target_agent):
    # load json and create model
    file_name = './src/reasoning/trainedmodels/'+\
     env_name+'_'+target_agent+'_model'
    json_file = open(file_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(file_name+".h5")
    return loaded_model

def save_data(exp_replay_buffer,env_name,target_agent_index):
    file_name = './src/reasoning/trainedmodels/'+\
         env_name+'_'+target_agent_index+'.pickle'
    with open(file_name, 'wb') as handle:
        pickle.dump(exp_replay_buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_data(env_name,target_agent_index):
    exp_replay_buffer = []
    file_name = './src/reasoning/trainedmodels/'+\
        env_name+'_'+target_agent_index+'.pickle'
    if os.path.exists(file_name):
        with open(file_name, 'rb') as handle:
            exp_replay_buffer = pickle.load(handle)
        print('Loaded! Replay Buffer Size =',len(exp_replay_buffer))
    else:
        print('Load fail...')
    return exp_replay_buffer

class TrainingModel:

    def __init__(self,env_name : str, agent_index : str,
        max_epoch : int = 1000, discount_factor : float = 0.95):
        self.target_agent_index = agent_index

        self.env_name = env_name
        if 'LevelForagingEnv' in env_name:
            from src.envs.LevelForagingEnv import load_default_scenario
            scenario_id = int(list(env_name)[-1])
            env, scenario_id = load_default_scenario(\
                'mcts',scenario_id=scenario_id,display=False)
            print(env_name+' loaded.')
        else:
            raise NotImplemented
        
        self.env = env
        self.env.reset()
        self.max_epoch = max_epoch
        self.discount_factor = discount_factor
        self.epsilon = 0.99
        self.epsilon_decay = 0.98/self.max_epoch
        self.exp_replay_buffer = []
        self.qmodel = None

    def step(self):
        prev_state = self.env.get_rlmodel_state()
        state,reward,done,info = self.env.step(self.env.action_space.sample())
        target_agent = self.env.get_agent_by_index(self.target_agent_index)
        action = target_agent.next_action
        current_state = self.env.get_rlmodel_state()
        return [[prev_state,action,reward,current_state],done]
    
    def init_exp_replay_buffer(self, max_iterations: int = 1000):
        # load replay buffer, if exists
        self.exp_replay_buffer = load_data(self.env_name,self.target_agent_index)
        if len(self.exp_replay_buffer) >= max_iterations:
            return 

        # creating
        update_time = time()
        for it in range(max_iterations):
            if (time() - update_time) > 10:
                print('Replay buffer size {:d} - Progress {:.2f} %'\
                    .format(len(self.exp_replay_buffer),100*it/max_iterations),end='\r')
                update_time = time()
            
            # stepping and saving experience
            if len(self.exp_replay_buffer) < max_iterations:
                self.exp_replay_buffer.append(self.step())
            else:
                self.exp_replay_buffer.pop(0)
                self.exp_replay_buffer.append(self.step())
            
            
            if self.exp_replay_buffer[-1][1] == True:
                print('===== Resetting the environment.',it,'======')
                self.env.reset()

        save_data(self.exp_replay_buffer,self.env_name,self.target_agent_index)
        return


    def get_qlearning_batch(self, model, batch_size):
        memory_length = len(self.exp_replay_buffer)
        inputs = np.zeros((min(memory_length,batch_size),self.env.get_rlmodel_input_shape()))

        # Placeholder for target q-values after applying model to state input
        num_inputs = inputs.shape[0] # vector/array
        targets = np.zeros((num_inputs, len(self.env.actions)))

        # For each element in the batch of size batch_size randomly select an experience
        # save it into inputs, and calculate the q-values for state to be saved into
        # targets.
        ids = np.random.choice(memory_length, size=batch_size, replace=False)

        # Select random experience
        sars = [self.exp_replay_buffer[id_][0] for id_ in ids]
        previous_states = [list(np.array(e[0]).flatten()) for e in sars]
        action_ts       = [e[1] for e in sars]
        rewards         = [e[2] for e in sars]
        current_states  = [list(np.array(e[3]).flatten()) for e in sars]

        # Use state to calculate q-values for each action
        targets = model.predict(previous_states, batch_size=64, verbose=False)

        # Greedily choose maximum q-value
        Q_sa = np.max(model.predict(current_states, batch_size=64, verbose=False), 1)

        # Save into targets
        # reward + gamma * max_a' Q(s', a')
        for i in range(batch_size):
            targets[i, action_ts[i]] = (
                rewards[i] + self.discount_factor * Q_sa[i] 
            )

        return previous_states, targets
    
    def get_qmodel(self,input_shape, hidden_size : int, num_actions : int, learning_rate : float = 0.01,
     hidden_activation : str = "relu", loss : str = "mse", hidden_layers : int = 2) -> Type[keras.Model]:

        qmodel = Sequential()
        qmodel.add(Dense(hidden_size, input_shape=input_shape,activation=hidden_activation))
        for _ in range(hidden_layers - 1):
            qmodel.add(Dense(hidden_size, activation=hidden_activation))
        qmodel.add(Dense(num_actions))
        qmodel.compile(SGD(learning_rate=learning_rate), loss)
        print(learning_rate)
        return qmodel

    def train(self, max_iterations : int, batch_size : int):
        if model_exists(self.env_name,self.target_agent_index):
            print('Model weights and archtecure loaded!')
            return load_model(self.env_name,self.target_agent_index)
        
        self.init_exp_replay_buffer(max_iterations)
        model = self.get_qmodel(input_shape=(self.env.get_rlmodel_input_shape(),),
            hidden_size=256, num_actions=len(self.env.actions),hidden_layers=2)

        # Training loop
        loss = 0.0

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x, y = [], []
        
        line, = ax.plot(x,y)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')

        for epoch in range(self.max_epoch):
            inputs, targets = self.get_qlearning_batch(model, batch_size=batch_size)
            loss += model.train_on_batch(np.array(inputs), np.array(targets))

            x.append(epoch)
            line.set_xdata(x)

            y.append(loss)
            line.set_ydata(y)

            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()

            if epoch % (0.2*self.max_epoch) == 0:
                print('Saving partial model. Progress:', 100 * epoch / (self.max_epoch))
                save_model(model,self.env_name,self.target_agent_index)

        return model