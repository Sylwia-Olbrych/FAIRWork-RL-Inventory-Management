import numpy as np
from warehouse_env import InvOptEnv
from seasonal_demand import load_demand_records
import matplotlib.pyplot as plt

class FixedQuantityHeuristic:
    def __init__(self, fixed_quantity):
        self.fixed_quantity = fixed_quantity

    def choose_action(self, observation):
        return self.fixed_quantity

class MaintainInventoryLevelHeuristic:
    def __init__(self, target_level):
        self.target_level = target_level

    def choose_action(self, observation):
        current_inventory = np.atleast_1d(observation[0])[0]
        order_quantity = np.maximum(0, self.target_level - current_inventory)
        return order_quantity

# Load demand records
all_data_sets = []
for seed in range(51, 61):
    demand_record = load_demand_records(seed)
    all_data_sets.append(demand_record)

# Create the heuristics
heuristics = [
    FixedQuantityHeuristic(fixed_quantity=6),
    MaintainInventoryLevelHeuristic(target_level=35)
]

# Create lists to store the rewards for each heuristic
fixed_rewards = []  # type: ignore
maintain_inventory_level_rewards = []   # type: ignore

# Lists to store data for plotting
demand_records_plot = []

# ...

# Simulate the environment using each heuristic
for i, data_set in enumerate(all_data_sets):
    # Initialize the environment with the current data set
    env = InvOptEnv(data_set)

    # Lists to store data for plotting
    demand_records_plot.append(data_set)
    action_plot = []    # type: ignore
    inventory_level_plot = []   # type: ignore

    for heuristic in heuristics:
        state = env.reset()  # Reset the environment for each heuristic

        actions_list = []  # Initialize an empty list to store the actions
        inv_list = []  # Initialize an empty list to store the obs[0] values
        counter = 0  # Initialize the counter variable
        total_reward = 0  # Initialize the total reward variable
        done = False  # Reset the done variable

        while not done:
            action = heuristic.choose_action(state)  # type: ignore
            next_state, reward, terminate, _, _ = env.step(action)

            actions_list.append(action)  # Store the current action in the actions list
            inv_list.append(next_state[0])
            total_reward += reward  # Accumulate the reward at each step
            counter += 1

            if terminate or env.current_period >= len(data_set):  # Check if we reached the end of demand records
                break

            state = next_state

        # Plotting
        plt.figure(figsize=(10, 8))

        # Plot the demand record
        plt.subplot(3, 1, 1)
        plt.plot(range(len(data_set)), data_set, label='Demand Record', color='green')
        plt.xlabel('Time Step')
        plt.ylabel('Demand')
        plt.title('Demand Record Over Time')
        plt.legend()

        # Plot the action
        plt.subplot(3, 1, 2)
        plt.plot(range(counter), actions_list, label=f"{heuristic.__class__.__name__} Action")
        plt.xlabel('Time Step')
        plt.ylabel('Action Value')
        plt.title('Action Over Time')
        plt.legend()

        # Plot the inventory level
        plt.subplot(3, 1, 3)
        plt.plot(range(counter), inv_list, label='Inventory Level', color='orange')
        plt.xlabel('Time Step')
        plt.ylabel('Inventory Level')
        plt.title('Inventory Level Over Time')
        plt.legend()

        plt.tight_layout()
        plt.show()

        print(f"Total reward of data set {i + 1} ({heuristic.__class__.__name__}): {total_reward}")
        # print("Total Steps:", counter)
