import numpy as np
import matplotlib.pyplot as plt


def load_demand_records(seed=None):
    """Load the demand records with a specified seed."""
    if seed is None:
        seed = np.random.randint(0, 100)
    np.random.seed(seed)
    demand_hist = []

    # Winter (lower demand)
    for _ in range(13):
        random_demand = np.random.normal(3, 1)
        if random_demand < 0:
            random_demand = 0
        random_demand = np.round(random_demand)
        demand_hist.append(random_demand)

    # Spring (moderate demand)
    for _ in range(13):
        random_demand = np.random.normal(8, 1)
        if random_demand < 0:
            random_demand = 0
        random_demand = np.round(random_demand)
        demand_hist.append(random_demand)

    # Summer (higher demand)
    for _ in range(13):
        random_demand = np.random.normal(15, 1)
        if random_demand < 0:
            random_demand = 0
        random_demand = np.round(random_demand)
        demand_hist.append(random_demand)

    # Fall (moderate demand)
    for _ in range(13):
        random_demand = np.random.normal(7, 1)
        if random_demand < 0:
            random_demand = 0
        random_demand = np.round(random_demand)
        demand_hist.append(random_demand)

    return demand_hist


demand_record = load_demand_records()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(demand_record)
plt.title("Demand Record for One Year")
plt.xlabel("Week")
plt.ylabel("Demand")
plt.grid(True)
plt.show()
