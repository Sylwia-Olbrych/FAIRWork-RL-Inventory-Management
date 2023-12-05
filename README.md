Developers: Zi Xuan (Melody) Tung, Alexander Nasuta, Sylwia Olbrych

# Warehouse Optimization with Reinforcement Learning

## Introduction
This repository contains code for an inventory optimization project that uses Reinforcement Learning to manage inventory levels in response to seasonal demand fluctuations. The project employs the [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) library and leverages [Weights & Biases (WandB)](https://wandb.ai/) for hyperparameter tuning and logging experiment results.

## Project Description
The goal of this project is to develop a reinforcement learning agent that optimizes inventory levels in a warehouse to minimize costs while satisfying customer demand. The project simulates different demand patterns and uses Proximal Policy Optimization (PPO) to train the agent to make inventory management decisions.

## Files in the Repository
- `seasonal_demand.py`: Generates seasonal demand patterns for training and evaluation.
- `warehouse_env.py`: Defines the Gym environment for inventory optimization.
- `warehouse_sweep.py`: Conducts hyperparameter sweep using WandB. 
- `warehouse_best_sweep.py`: The best run from hyperparameter sweep is further trained to obtain trained model.
- `warehouse_restore.py`: Restore the trained model from WandB and tests it with different demand patterns.
- `warehouse_local_model`: Use the saved model from restored from `warehouse_restore.py` locally. 
- `heuristics.py`: Provides two different heuristics to compare the result of the reinforcement learning model to.
## Getting Started
1. Clone the repository to your local machine.
2. Install the required Python packages and libraries by running `pip install -r requirements.txt`.
3. Ensure you have WandB credentials set up for tracking and monitoring experiments.
4. Run these scripts in this order:

   1. Hyperparameter Tuning: `warehouse_sweep.py` - The script will log the hyperparameter sweep details to WandB.
   2. Further Training: `warehouse_best_sweep.py` - The script will continue training the best run from the sweep. You can find the id of the best run in WandB and enter the id in run_id. 
   3. Model Testing: `warehouse_restore.py` - The script restores the model after training is finished and saves model locally to desired file. The script will test the model on different demand patterns and log the results, including total rewards and inventory levels for evaluation.

   
## Results
The results of the experiments, including hyperparameter tuning and model evaluations, are logged and visualized using WandB. You can view the latest results of the latest run on the WandB project page (https://wandb.ai/team-friendship/warehouse-sweep-v23?workspace=user-zi-xuan-tung).

## Development 
The following sections are only relevant if you plan on further develop the environment and introduce code changes into 
the environment itself.

To run this Project locally on your machine follow the following steps:

1. Clone the repo
   ```sh
   git clone https://github.com/Sylwia-Olbrych/FAIRWork-RL-Inventory-Management.git
   ```
2. Install the python requirements_dev packages. `requirements_dev.txt` includes all the packages of
specified `requirements.txt` and some additional development packages like `mypy`, `pytext`, `tox` etc. 
    ```sh
   pip install -r requirements_dev.txt
   ```
3. Install the modules of the project locally. For more info have a look at 
[James Murphy's testing guide](https://www.youtube.com/watch?v=DhUpxWjOhME)
   ```sh
   pip install -e .
   ```

### Testing

For testing make sure that the dev dependencies are installed (`requirements_dev.txt`) and the models of this 
project are set up (i.e. you have run `pip install -e .`).  

Then you should be able to run

```sh
mypy src
```

```sh
flake8 src
```

```sh
pytest
```

or everthing at once using `tox`.

```sh
tox
```


“This work has been supported by the FAIRWork project (www.fairwork-project.eu) and has been funded within the European Commission’s Horizon Europe Programme under contract number 101049499. This paper expresses the opinions of the authors and not necessarily those of the European Commission. The European Commission is not liable for any use that may be made of the information contained in this presentation.”

## Copyright
Copyright © RWTH of FAIRWork Consortium
