Developers: Zi Xuan(Melody) Tung, Alexander Nasuta, Sylwia Olbrych

# Warehouse Optimization with Reinforcement Learning

## Introduction
This repository contains code for an inventory optimization project that uses Reinforcement Learning to manage inventory levels in response to seasonal demand fluctuations. The project employs the [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) library and leverages [Weights & Biases (WandB)](https://wandb.ai/) for hyperparameter tuning and logging experiment results.

## Project Description
The goal of this project is to develop a reinforcement learning agent that optimizes inventory levels in a warehouse to minimize costs while satisfying customer demand. The project simulates different demand patterns and uses Proximal Policy Optimization (PPO) to train the agent to make inventory management decisions.

## Files in the Repository
- `demand_records.py`: Generates synthetic demand records for training and evaluation.
- `seasonal_demand.py`: Generates seasonal demand patterns for training and evaluation.
- `warehouse_env.py`: Defines the Gym environment for inventory optimization.
- `warehouse_train.py`: Trains the PPO model with hyperparameter tuning.
- `warehouse_solve.py`: Conducts hyperparameter tuning experiments using WandB.
- `warehouse_restore.py`: Tests the trained model with different demand patterns.
- `README.md`: This file.

## Getting Started
1. Clone the repository to your local machine.
2. Install the required Python packages and libraries by running `pip install -r requirements.txt`.
3. Ensure you have WandB credentials set up for tracking and monitoring experiments.
4. To run the training, testing, or hyperparameter tuning scripts, use the following commands:

    - Training: `python warehouse_train.py`
    - Hyperparameter Tuning: `python warehouse_solve.py`
    - Model Testing: `python warehouse_restore.py`

## Training the Model
1. Run `python warehouse_train.py` to train the PPO model with hyperparameter tuning.
2. The training script will generate a model and log training details to WandB.

## Evaluating the Model
1. After training, you can evaluate the model's performance by running `python warehouse_restore.py`.
2. The script will test the model on different demand patterns and log the results, including total rewards and inventory levels.

## Results
The results of the experiments, including hyperparameter tuning and model evaluations, are logged and visualized using WandB. You can view the results on the WandB project page.

“This work has been supported by the FAIRWork project (www.fairwork-project.eu) and has been funded within the European Commission’s Horizon Europe Programme under contract number 101049499. This paper expresses the opinions of the authors and not necessarily those of the European Commission. The European Commission is not liable for any use that may be made of the information contained in this presentation.”

## Copyright
Copyright © RWTH of FAIRWork Consortium
