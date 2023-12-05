# This script is designed to load trained model and test model with test set

import wandb
from warehouse_env import InvOptEnv
from seasonal_demand import load_demand_records
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

# Define the username, project name, and run ID of the run where the model is saved
username = 'team-friendship'
project = 'warehouse-sweep-v23'
run_id = 'qtkyuemg'

# Initialize W&B for the run where the model is saved
wandb.init(entity=username, project=project, id=run_id)
model_filename = "model_bulkdisc.h5"

# Restore the model file "model.h5" from the specified run
best_model = wandb.restore('model_seasonal_extended.h5', run_path=f"{username}/{project}/{run_id}")

# Create the PPO model with the same configuration used during training
model = PPO.load(best_model.name)   # type: ignore

# Get the absolute path to the directory where you want to save the model
model_save_dir = os.getcwd()  #"/src"
local_model_filename = "model_v23"

# Create the directory if it does not exist
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# Save the model to the specified directory
model.save(os.path.join(model_save_dir, local_model_filename))


# # Get the absolute path to the model file
# model_file_path = os.path.join(model_save_dir, local_model_filename)
#
# # Check if the model file exists
# if os.path.isfile(model_file_path):
#     # Load the model
#     model = PPO.load(model_file_path)
# else:
#     # The model file does not exist
#     print("The model file does not exist")


all_data_sets = []
for seed in range(51, 61):  # This loop will include seeds from 51 to 60
    demand_record = load_demand_records(seed)
    all_data_sets.append(demand_record)

for i, data_set in enumerate(all_data_sets):
    # Initialize the environment with the current data set

    env = DummyVecEnv([lambda: InvOptEnv(data_set)])
    obs = env.reset()
    done = False

    actions_list = []  # Initialize an empty list to store the actions
    acc_rewards_list = []
    inv_list = []  # Initialize an empty list to store the obs[0] values
    counter = 0  # Initialize the counter variable
    total_reward = 0  # Initialize the total reward variable

    while not done:
        action, _ = model.predict(obs)  # type: ignore
        actions_list.append(action)  # Store the current action in the actions list
        obs, reward, done, _ = env.step(action)  # type: ignore
        inv_list.append(obs[0])  # type: ignore # Store the obs[0] value in the obs_values list
        acc_rewards_list.append(total_reward)  # Store the accumulated reward at this step

        total_reward += reward  # type: ignore

        # Increment the counter
        counter += 1
    print(f"Total reward of data set {i + 1} : {total_reward}")
    # print("Total Steps:", counter)
