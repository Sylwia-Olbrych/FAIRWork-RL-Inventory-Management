import gymnasium
import numpy as np
from gymnasium.spaces import Box
from gymnasium.envs.registration import register
from seasonal_demand import load_demand_records  # , convert_day_to_month_fraction
from gym import Env
from typing import TypeVar


# Load demand records
demand_records = load_demand_records()

# State = TypeVar('State')
# Action = TypeVar('Action')
#
# class InvOptEnv(gymnasium.Env[State, Action]):
#     metadata = {'render.modes': ['human']}


class InvOptEnv(gymnasium.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, demand_records):
        super(InvOptEnv, self).__init__()
        self.n_period = len(demand_records)
        self.n_months = 12
        self.current_period = 1
        self.day_of_week = 0
        self.capacity = 100
        self.inv_level = self.capacity * 0.5
        # self.inv_pos = self.capacity * 0.5
        self.holding_cost = 3
        self.unit_price = 30
        self.fixed_order_cost = 50          # e.g. delivery costs
        self.variable_order_cost = 10       # cost per unit
        self.shortage_cost = 10
        self.lead_time = 2
        self.order_arrival_list = []
        self.demand_list = demand_records
        self.state = np.array([self.inv_level] + self.convert_day_to_month_fraction(self.day_of_week))
        self.state_list = []
        self.state_list.append(self.state)
        self.action_list = []
        self.reward_list = []

        obs_low = np.array([0, 0], dtype=np.float32)  # np.array([0] * (1 + 6), dtype=np.float32)
        obs_high = np.array([self.capacity, self.n_months],
                            dtype=np.float32)  # np.array([self.capacity] + [1] * 6, dtype=np.float32)
        self.action_space = Box(low=0, high=self.capacity, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state_list = []
        self.action_list = []
        self.reward_list = []
        self.inv_level = self.capacity * 0.5
        # self.inv_pos = self.capacity * 0.5
        self.current_period = 1
        self.day_of_week = 0
        self.state = np.array([self.inv_level] + self.convert_day_to_month_fraction(self.current_period),
                              dtype=np.float32)
        self.state_list.append(self.state)
        self.order_arrival_list = []
        self.seed(seed)
        # self.current_obs = np.array([self.inv_pos] + self.convert_day_of_week(self.day_of_week))
        return self.state, {}

    def step(self, action):
        discount_rate = 0
        # action = np.ceil(action)
        # action = max(0, min(self.capacity, action))
        if action > 0:
            self.order_arrival_list.append([self.current_period + self.lead_time, action])

        if len(self.order_arrival_list) > 0:
            if self.current_period == self.order_arrival_list[0][0]:
                self.inv_level = min(self.capacity, self.inv_level + self.order_arrival_list[0][1])
                self.order_arrival_list.pop(0)
        # print("Current period", self.current_period)
        demand = self.demand_list[self.current_period - 1]
        # order_arrival = self.order_arrival_list
        units_sold = demand if demand <= self.inv_level else self.inv_level

        # penalty when inventory is less than 40% -> newest update: 10%
        inventory_penalty = 0
        if self.inv_level < 0.4 * self.capacity:
            inventory_penalty = -100  # Set a negative penalty value  - 0 * self.fixed_order_cost \

        # Apply bulk discount based on the action value

        # if 10 <= action <= 50:
        #     discount_rate = 0.2 + 0.2 * (action - 10) / 40
        #     # cost *= (1 - discount_rate)

        reward = units_sold * self.unit_price - self.holding_cost * self.inv_level \
            - action * self.variable_order_cost + inventory_penalty

        # Calculate shortage penalty
        if demand > self.inv_level:
            shortage_penalty = (demand - self.inv_level) * self.shortage_cost
            reward -= shortage_penalty
        reward = float(reward)

        # Update inventory level
        self.inv_level = max(0, self.inv_level - units_sold)
        # self.inv_pos = self.inv_level

        # Update inventory level based on incoming orders
        if len(self.order_arrival_list) > 0:
            for i in range(len(self.order_arrival_list)):
                # Update inventory based on incoming orders
                self.inv_level += self.order_arrival_list[i][1]
                # Ensure inv_level does not exceed capacity
                self.inv_level = min(self.inv_level, self.capacity)

        # Update state
        self.state = np.array([float(self.inv_level)] + self.convert_day_to_month_fraction(self.current_period),
                              dtype=np.float32)
        self.current_period += 1
        self.state_list.append(self.state)
        self.action_list.append(action)
        self.reward_list.append(reward)

        # info = {}
        from typing import Dict

        info: Dict[type, type] = {}

        # Check if episode is complete
        if self.current_period > self.n_period:
            terminate = True
            truncate = True
        else:
            terminate = False
            truncate = False
        # self.current_obs = self.state
        return self.state, reward, terminate, truncate, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        return

    def convert_day_to_month_fraction(self, day):
        month_lengths = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        total_days = 0
        month = 1
        for month_length in month_lengths:
            if day <= total_days + month_length:
                # fraction = (day - total_days) / month_length
                return [month]
            total_days += month_length
            month += 1


register(id='Inv-v1', entry_point='warehouse_env:InvOptEnv', kwargs={'demand_records': demand_records})


from gymnasium.utils.env_checker import check_env
# from inventory.seasonal_demand import load_demand_records #, convert_day_to_month_fraction

# Load demand records
demand_records = load_demand_records()

    # Create an instance of your custom environment with the required arguments
# env = InvOptEnv(demand_records=demand_records)
    # env = DummyVecEnv([lambda: env])  # Wrap the environment in DummyVecEnv if needed
from typing import Annotated
env: Annotated[InvOptEnv, "This is a type annotation for env"]
env = InvOptEnv(demand_records=demand_records)

check_env(env)

