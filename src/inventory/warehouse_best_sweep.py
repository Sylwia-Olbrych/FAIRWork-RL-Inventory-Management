# This file loads the best parameters ( from wandb ) and lets users train the agent with a longer timestep
# Paste the run ID from wandb for the set of parameters that we want and train the agent with more timesteps.

import wandb
import os
from warehouse_env import InvOptEnv
from seasonal_demand import load_demand_records
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
# Custom callback for tracking timesteps and logging to wandb


class CustomCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.total_timesteps = 0
        self.best_mean_reward = float("-inf")
        self.best_timestep = 0

    def _on_step(self) -> bool:
        # Increment the total number of timesteps
        self.total_timesteps += 1

        # Get the current timestep
        current_timestep = self.num_timesteps

        # Check if the current mean reward is better than the best mean reward
        if eval_callback.best_mean_reward > self.best_mean_reward:
            self.best_mean_reward = eval_callback.best_mean_reward
            self.best_timestep = current_timestep

        return True  # Continue training

    def _on_training_end(self) -> None:
        # Log the total number of timesteps, best mean reward,
        # and the timestep it occurred at
        wandb.log({
            "Total Timesteps": self.total_timesteps,
            "Best Mean Reward": self.best_mean_reward,
            "Best Mean Reward at Timestep": self.best_timestep
        })

# Define the username, project name, and run ID of the best run


username = 'team-friendship'
project = 'warehouse-sweep-v23'
run_id = 'qtkyuemg'

# Initialize WandB for the best run
wandb.init(entity=username, project=project, id=run_id, resume=True)

# Load the best model's configuration
# config = wandb.config
# print(config)
# learning_rate = wandb.config.get('learning_rate')  # Use .get() to avoid KeyError
# print(learning_rate)
# n_epochs = wandb.config.get('n_epochs')
# print(n_epochs)
# gae_lambda = wandb.config.get('gae_lambda')
# print(gae_lambda)
#
# # # Load the best model's configuration
config = wandb.config
# print(config)
learning_rate = config.learning_rate
n_epochs = config.n_epochs
gae_lambda = config.gae_lambda

# Generate demand records for all seeds
all_data_sets = []
for seed in range(50):
    demand_record = load_demand_records(seed)
    all_data_sets.append(demand_record)

for i, data_set in enumerate(all_data_sets):
    # Initialize the environment with the current data set
    print(f"Processing data set {i + 1} of {len(all_data_sets)}")
    env = DummyVecEnv([lambda: InvOptEnv(data_set)])

    # Create the PPO model using the best hyperparameters
    # model = PPO('MlpPolicy', env, learning_rate=learning_rate, verbose=1, n_epochs=n_epochs,
    #             gae_lambda=gae_lambda)
    model = PPO('MlpPolicy', env, learning_rate=learning_rate, verbose=1, n_epochs=n_epochs,
                gae_lambda=gae_lambda)

    # Load the best model's weights
    # model.load(os.path.join(wandb.run.dir, "model_seasonal.h5"))

    # Continue training with more timesteps
    total_timesteps = int(1e6)  # Adjust to your desired number of additional timesteps

    custom_callback = CustomCallback()

    # Evaluation environment -* has to be wrapped in the same way as training environment
    # (replace with evaluation data)
    eval_env = DummyVecEnv([lambda: InvOptEnv(load_demand_records(70))])

    # Add EvalCallback for periodic evaluation and logging
    eval_callback = EvalCallback(eval_env, callback_on_new_best=custom_callback,
                                 best_model_save_path=os.path.join(wandb.run.dir, "best_model"),       # type: ignore
                                 log_path=wandb.run.dir, eval_freq=100000, verbose=1)   # type: ignore
    wandb.log({'best_mean_reward': eval_callback.best_mean_reward})

    model.learn(total_timesteps, callback=[custom_callback, eval_callback])

# Evaluate the model if needed
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=20)

# Log metrics to WandB
wandb.log({'mean_reward': mean_reward})

# Save the updated model
model.save(os.path.join(wandb.run.dir, "model_seasonal_extended.h5"))   # type: ignore


# Finish the WandB run
wandb.run.finish()  # type: ignore
