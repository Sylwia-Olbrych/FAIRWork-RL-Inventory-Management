# This script is designed to load trained model and test model
# with test set

import wandb
from warehouse_env import InvOptEnv
from seasonal_demand import load_demand_records
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv



# Define the username, project name, and run ID of the run where the model is saved
username = 'team-friendship'
project = 'warehouse-sweep-v18-seasonal-set'
run_id = 'k1abvzso' # '39ea3gcx'

# Initialize W&B for the run where the model is saved
wandb.init(entity=username, project=project, id=run_id)
model_filename = "model_seasonal_extended.h5"

# Restore the model file "model.h5" from the specified run
best_model = wandb.restore('model_seasonal_extended.h5', run_path=f"{username}/{project}/{run_id}")


# Create the PPO model with the same configuration used during training
model = PPO.load(best_model.name)


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
        action, _ = model.predict(obs)
        actions_list.append(action)  # Store the current action in the actions list
        obs, reward, done, _ = env.step(action)
        inv_list.append(obs[0])  # Store the obs[0] value in the obs_values list
        acc_rewards_list.append(total_reward)  # Store the accumulated reward at this step

        total_reward += reward  # Accumulate the reward at each step

        # Increment the counter
        counter += 1
    print(f"Total reward of data set {i + 1} : {total_reward}")
    # print("Total Steps:", counter)




