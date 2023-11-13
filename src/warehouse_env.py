import matplotlib.pyplot as plt
import gym
import numpy as np
import math
from gym.spaces import Box
from numpy.random import default_rng
from gym.envs.registration import register
from demand_records import load_demand_records

# Load demand records
demand_records = load_demand_records()

class InvOptEnv(gym.Env):
    def __init__(self, demand_records):
        self.n_period = len(demand_records)
        self.current_period = 1
        self.day_of_week = 0
        self.inv_level = 25
        self.inv_pos = 25
        self.capacity = 50
        self.holding_cost = 3
        self.unit_price = 30
        self.fixed_order_cost = 50
        self.variable_order_cost = 10
        self.shortage_cost = 2
        self.lead_time = 2
        self.order_arrival_list = []
        self.demand_list = demand_records
        self.state = np.array([self.inv_pos] + self.convert_day_of_week(self.day_of_week))
        self.state_list = []
        self.state_list.append(self.state)
        self.action_list = []
        self.action_space = Box(low=0, high=self.capacity, shape=(1,), dtype=np.float32)
        self.reward_list = []

        obs_low = np.array([0] * (1 + 6), dtype=np.float32)
        obs_high = np.array([self.capacity] + [1] * 6, dtype=np.float32)
        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)

    def reset(self):
        self.state_list = []
        self.action_list = []
        self.reward_list = []
        self.inv_level = 25
        self.inv_pos = 25
        self.current_period = 1
        self.day_of_week = 0
        self.state = np.array([self.inv_pos] + self.convert_day_of_week(self.day_of_week), dtype=np.float32)
        self.state_list.append(self.state)
        self.order_arrival_list = []
        # self.current_obs = np.array([self.inv_pos] + self.convert_day_of_week(self.day_of_week))

        return self.state

    def step(self, action):
        # action = np.ceil(action)
        # action = max(0, min(self.capacity, action))
        if action > 0:
            y = 1
            self.order_arrival_list.append([self.current_period + self.lead_time, action])
        else:
            y = 0
        if len(self.order_arrival_list) > 0:
            if self.current_period == self.order_arrival_list[0][0]:
                self.inv_level = min(self.capacity, self.inv_level + self.order_arrival_list[0][1])
                self.order_arrival_list.pop(0)
        # print("Current period", self.current_period)
        demand = self.demand_list[self.current_period - 1]
        # print("Current demand:", demand)
        order_arrival = self.order_arrival_list
        # print("Order arrival list:", order_arrival)
        units_sold = demand if demand <= self.inv_level else self.inv_level

        #penalty when inventory is less than 40%
        inventory_penalty = 0
        if self.inv_level < 0.4 * self.capacity:
            inventory_penalty = -100  # Set a negative penalty value  - 0 * self.fixed_order_cost \

        reward = units_sold * self.unit_price - self.holding_cost * self.inv_level \
                - action * self.variable_order_cost + inventory_penalty

        # reward = units_sold * self.unit_price - self.holding_cost * self.inv_level - y * self.fixed_order_cost \
        #          - action * self.variable_order_cost
        if demand > self.inv_level:
            shortage_penalty = (demand - self.inv_level) * self.shortage_cost
            reward -= shortage_penalty
        reward = float(reward)
        self.inv_level = max(0, self.inv_level - units_sold)
        self.inv_pos = self.inv_level
        if len(self.order_arrival_list) > 0:
            for i in range(len(self.order_arrival_list)):
                self.inv_pos += self.order_arrival_list[i][1]
        # if len(self.order_arrival_list) > 0:
        #     for i in range(len(self.order_arrival_list)):
        #         self.inv_pos += self.order_arrival_list[i][1]
        #         self.inv_pos = min(self.inv_pos, self.capacity)  # Ensure inv_pos does not exceed capacity

        self.day_of_week = (self.day_of_week + 1) % 7
        # self.state = np.array([self.inv_pos] + self.convert_day_of_week(self.day_of_week), dtype=np.float32)
        self.state = np.array([float(self.inv_level)] + self.convert_day_of_week(self.day_of_week), dtype=np.float32)
        # print("inv_pos:", self.inv_pos)
        # print("converted day of week:", self.convert_day_of_week(self.day_of_week))

        self.current_period += 1
        self.state_list.append(self.state)
        self.action_list.append(action)
        self.reward_list.append(reward)
        if self.current_period > self.n_period:
            terminate = True
        else:
            terminate = False
        # self.current_obs = self.state
        return self.state, reward, terminate, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        return

    def convert_day_of_week(self, d):
            if d == 0:
                return [0, 0, 0, 0, 0, 0]
            if d == 1:
                return [1, 0, 0, 0, 0, 0]
            if d == 2:
                return [0, 1, 0, 0, 0, 0]
            if d == 3:
                return [0, 0, 1, 0, 0, 0]
            if d == 4:
                return [0, 0, 0, 1, 0, 0]
            if d == 5:
                return [0, 0, 0, 0, 1, 0]
            if d == 6:
                return [0, 0, 0, 0, 0, 1]


register(id='Inv-v1', entry_point='warehouse_env:InvOptEnv', kwargs={'demand_records': demand_records})