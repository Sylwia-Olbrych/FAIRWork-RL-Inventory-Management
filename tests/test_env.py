from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.utils.env_checker import check_env
from inventory.warehouse_env import InvOptEnv  # Make sure to use the correct import statement
from inventory.seasonal_demand import load_demand_records #, convert_day_to_month_fraction


def test_env():
    # Load demand records
    demand_records = load_demand_records()

    # Create an instance of your custom environment with the required arguments
    env = InvOptEnv(demand_records=demand_records)
    # env = DummyVecEnv([lambda: env])  # Wrap the environment in DummyVecEnv if needed

    # Check the environment
    check_env(env)
