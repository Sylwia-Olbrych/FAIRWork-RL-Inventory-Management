from warehouse_env import InvOptEnv
from seasonal_demand import load_demand_records
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

# Get the absolute path to the directory where you want to save the model
model_save_dir = "/Users/melodytung/PycharmProjects/FAIRWork-RL-Inventory-Management/src/inventory"
local_model_filename = "model_v23"

# Get the absolute path to the model file
model_file_path = os.path.join(model_save_dir, local_model_filename)

# Load the model
model = PPO.load(model_file_path)

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