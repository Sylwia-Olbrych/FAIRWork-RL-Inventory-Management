import gym
import numpy as np
from warehouse_env import InvOptEnv
from seasonal_demand import load_demand_records

class FixedQuantityHeuristic:
    def __init__(self, fixed_quantity):
        self.fixed_quantity = fixed_quantity

    def choose_action(self, observation):
        return self.fixed_quantity

class MaintainInventoryLevelHeuristic:
    def __init__(self, target_level):
        self.target_level = target_level

    def choose_action(self, observation):
        current_inventory = observation[0]
        order_quantity = max(0, self.target_level - current_inventory)
        return order_quantity

class AverageDemandHeuristic:
    def __init__(self, demand_records):
        self.average_demand = np.mean(demand_records)

    def choose_action(self, observation):
        return self.average_demand

# Load demand records
all_data_sets = []
for seed in range(51, 61):  # This loop will include seeds from 51 to 60
    demand_record = load_demand_records(seed)
    all_data_sets.append(demand_record)

# Create the heuristics
heuristics = [
    FixedQuantityHeuristic(fixed_quantity=6),
    MaintainInventoryLevelHeuristic(target_level=35)
]

# Create lists to store the rewards for each heuristic
fixed_rewards = []
maintain_inventory_level_rewards = []

# Simulate the environment using each heuristic
for heuristic in heuristics:
    for data_set in all_data_sets:
        # Initialize the environment with the current data set
        env = InvOptEnv(data_set)
        state = env.reset()
        done = False

        total_reward = 0

        while not done:
            action = heuristic.choose_action(state)
            print(action)
            next_state, reward, terminate, _ = env.step(action)
            print(next_state)

            total_reward += reward

            if terminate:
                break

            state = next_state
        print('done')

        # Add the total reward for the current data set to the corresponding list
        if heuristic.__class__.__name__ == "FixedQuantityHeuristic":
            fixed_rewards.append(total_reward)
        elif heuristic.__class__.__name__ == "MaintainInventoryLevelHeuristic":
            maintain_inventory_level_rewards.append(total_reward)

# Print the rewards for each heuristic
print("FixedQuantityHeuristic:")
for i, reward in enumerate(fixed_rewards):
    print(f"Total reward of data set {i + 1} : {reward}")

print("MaintainInventoryLevelHeuristic:")
for i, reward in enumerate(maintain_inventory_level_rewards):
    print(f"Total reward of data set {i + 1} : {reward}")
