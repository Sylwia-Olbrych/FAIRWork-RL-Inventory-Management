# This script is designed for conducting hyperparameter tuning experiments using the
# Weights & Biases (WandB) platform in the context of training a reinforcement
# learning agent for solving inventory optimization problems in seasonal demand scenarios.
#
# Run sweep with 100 runs and select best parameters to train agent in warehouse

import os
import wandb
import numpy as np
from warehouse_env import InvOptEnv
from seasonal_demand import load_demand_records
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import matplotlib.pyplot as plt

# Custom callback for tracking timesteps and logging to wandb
class CustomCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log the current total number of timesteps to Weights & Biases
        wandb.log({"Timesteps": self.num_timesteps})

        return True  # Continue training

# Define sweep config
sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep_bayes',
    'metric': {'goal': 'maximize', 'name': 'mean_reward'},
    'parameters': {
        'learning_rate': {'min': 5e-6, 'max': 0.003},
        # 'batch_size': {'values': [16, 32, 64]},
        'n_epochs': {'min': 3, 'max': 30},
        'gae_lambda': {'min': 0.8, 'max': 0.99}
        # 'gamma': {'min': 0.8, 'max': 0.9997}
        # 'ent_coef': {'values': [0.05, 0.1, 0.2]},
        # 'vf_coef': {'values': [0.8, 1.0]},
    },
    # 'early_terminate': {
    #         'type': 'hyperband',
    #         # 'min_iter': 5,  # Specify the iteration for the first bracket
    #         'max_iter': 50,  # Specify the maximum number of iterations
    #         's': 2,  # Specify the total number of brackets
    #
    # }
    # 'run_cap': 8
}

# Initialize sweep and give name ( optional actually )
sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project='warehouse-sweep-v18-seasonal-set'
)

# Generate demand records for all seeds
all_data_sets = []
for seed in range(50):
    demand_record = load_demand_records(seed)
    all_data_sets.append(demand_record)

def main():
    run = wandb.init()

    # Note that we define values from `wandb.config`
    # instead of defining hard values
    learning_rate = wandb.config.learning_rate
    n_epochs = wandb.config.n_epochs
    # gamma = wandb.config.gamma
    gae_lambda = wandb.config.gae_lambda
    # ent_coef = wandb.config.ent_coef
    # vf_coef = wandb.config.vf_coef

    for data_set in all_data_sets:
        # Initialize the environment with the current data set
        env = DummyVecEnv([lambda: InvOptEnv(data_set)])

        # Create the PPO model
        model = PPO('MlpPolicy', env, learning_rate=learning_rate, verbose=1, n_epochs=n_epochs,
                    gae_lambda=gae_lambda)  # , ent_coef=ent_coef, vf_coef=vf_coef)

        # Set the total number of timesteps for training
        total_timesteps = int(2e4)
        model.learn(total_timesteps)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=20)

    # Log metrics to WandB
    wandb.log({'mean_reward': mean_reward})

    # Save the trained model in wandb.run.dir
    model.save(os.path.join(wandb.run.dir, "model_seasonal.h5"))


# Start sweep job
wandb.agent(sweep_id, function=main, count=100)


