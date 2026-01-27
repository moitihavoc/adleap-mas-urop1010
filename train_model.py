from src.utils.train import TrainingModel, save_model

# 1. Setting training parameters
trainedmodels_path = './src/reasoning/trainedmodels/'
max_iterations = 1000
batch_size = 32
target_environment = 'LevelForagingEnv5'
target_agent = 'X'

# 2. Loading environment and Training model
training_model = TrainingModel(target_environment,target_agent)

# 3. Training Q-learning model and getting the trained model
qmodel = training_model.train(max_iterations,batch_size)

# 4. Save trained model weights and architecture
save_model(qmodel,training_model.env_name,training_model.target_agent_index)